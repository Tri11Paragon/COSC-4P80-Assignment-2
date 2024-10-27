#pragma once
/*
 *  Copyright (C) 2024  Brett Terpstra
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "blt/std/assert.h"

#ifndef COSC_4P80_ASSIGNMENT_2_LAYER_H
#define COSC_4P80_ASSIGNMENT_2_LAYER_H
    
    #include <blt/std/types.h>
    #include <assign2/initializers.h>
    #include "blt/iterator/zip.h"
    #include "blt/iterator/iterator.h"
    #include "global_magic.h"

namespace assign2
{
    class neuron_t
    {
            friend layer_t;
        public:
            // empty neuron for loading from a stream
            explicit neuron_t(weight_view weights, weight_view dw): dw(dw), weights(weights)
            {}
            
            // neuron with bias
            explicit neuron_t(weight_view weights, weight_view dw, Scalar bias): bias(bias), dw(dw), weights(weights)
            {}
            
            Scalar activate(const Scalar* inputs, function_t* act_func)
            {
                z = bias;
                for (auto [x, w] : blt::zip_iterator_container({inputs, inputs + weights.size()}, {weights.begin(), weights.end()}))
                    z += x * w;
                a = act_func->call(z);
                return a;
            }
            
            void back_prop(function_t* act, const std::vector<Scalar>& previous_outputs, Scalar next_error)
            {
                // delta for weights
                error = act->derivative(z) * next_error;
                db = learn_rate * error;
                BLT_ASSERT(previous_outputs.size() == dw.size());
                for (auto [prev_out, d_weight] : blt::zip(previous_outputs, dw))
                {
                    // dw
                    d_weight = -learn_rate * prev_out * error;
                }
            }
            
            void update()
            {
                for (auto [w, d] : blt::in_pairs(weights, dw))
                    w += d;
                bias += db;
            }
            
            template<typename OStream>
            OStream& serialize(OStream& stream)
            {
                stream << bias;
                for (auto d : weights)
                    stream << d;
            }
            
            template<typename IStream>
            IStream& deserialize(IStream& stream)
            {
                for (auto& d : blt::iterate(weights).rev())
                    stream >> d;
                stream >> bias;
            }
            
            void debug() const
            {
                std::cout << bias << " ";
            }
        
        private:
            float z = 0;
            float a = 0;
            float bias = 0;
            float db = 0;
            float error = 0;
            weight_view dw;
            weight_view weights;
    };
    
    class layer_t
    {
            friend network_t;
        public:
            template<typename WeightFunc, typename BiasFunc>
            layer_t(const blt::i32 in, const blt::i32 out, function_t* act_func, WeightFunc w, BiasFunc b):
                    in_size(in), out_size(out), layer_id(layer_id_counter++), act_func(act_func)
            {
                neurons.reserve(out_size);
                weights.preallocate(in_size * out_size);
                weight_derivatives.preallocate(in_size * out_size);
                for (blt::i32 i = 0; i < out_size; i++)
                {
                    auto weight = weights.allocate_view(in_size);
                    auto dw = weight_derivatives.allocate_view(in_size);
                    for (auto& v : weight)
                        v = w(i);
                    neurons.push_back(neuron_t{weight, dw, b(i)});
                }
            }
            
            const std::vector<Scalar>& call(const std::vector<Scalar>& in)
            {
                outputs.clear();
                outputs.reserve(out_size);
#if BLT_DEBUG_LEVEL > 0
                if (in.size() != in_size)
                    throw std::runtime_exception("Input vector doesn't match expected input size!");
#endif
                for (auto& n : neurons)
                    outputs.push_back(n.activate(in.data(), act_func));
                return outputs;
            }
            
            std::pair<Scalar, Scalar> back_prop(const std::vector<Scalar>& prev_layer_output,
                                                const std::variant<blt::ref<const std::vector<Scalar>>, blt::ref<const layer_t>>& data)
            {
                Scalar total_error = 0;
                Scalar total_derivative = 0;
                std::visit(blt::lambda_visitor{
                        // is provided if we are an output layer, contains output of this net (per neuron) and the expected output (per neuron)
                        [this, &prev_layer_output, &total_error, &total_derivative](const std::vector<Scalar>& expected) {
                            for (auto [i, n] : blt::enumerate(neurons))
                            {
                                auto d = outputs[i] - expected[i];
                                auto d2 = 0.5f * (d * d);
                                total_error += d2;
                                total_derivative += d;
                                n.back_prop(act_func, prev_layer_output, d);
                            }
                        },
                        // interior layer
                        [this, &prev_layer_output](const layer_t& layer) {
                            for (auto [i, n] : blt::enumerate(neurons))
                            {
                                Scalar w = 0;
                                // TODO: this is not efficient on the cache!
                                for (auto nn : layer.neurons)
                                    w += nn.error * nn.weights[i];
                                n.back_prop(act_func, prev_layer_output, w);
                            }
                        }
                }, data);
                return {total_error, total_derivative};
            }
            
            void update()
            {
                for (auto& n : neurons)
                    n.update();
            }
            
            template<typename OStream>
            OStream& serialize(OStream& stream)
            {
                for (auto d : neurons)
                    stream << d;
            }
            
            template<typename IStream>
            IStream& deserialize(IStream& stream)
            {
                for (auto& d : blt::iterate(neurons).rev())
                    stream >> d;
            }
            
            [[nodiscard]] inline blt::i32 get_in_size() const
            {
                return in_size;
            }
            
            [[nodiscard]] inline blt::i32 get_out_size() const
            {
                return out_size;
            }
            
            void debug() const
            {
                std::cout << "Bias: ";
                for (auto& v : neurons)
                    v.debug();
                std::cout << std::endl;
                weights.debug();
            }

#ifdef BLT_USE_GRAPHICS
            
            void render() const
            {
            
            }

#endif
        
        private:
            const blt::i32 in_size, out_size;
            const blt::size_t layer_id;
            weight_t weights;
            weight_t weight_derivatives;
            function_t* act_func;
            std::vector<neuron_t> neurons;
            std::vector<Scalar> outputs;
    };
}

#endif //COSC_4P80_ASSIGNMENT_2_LAYER_H
