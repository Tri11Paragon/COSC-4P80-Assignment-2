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

#ifndef COSC_4P80_ASSIGNMENT_2_LAYER_H
#define COSC_4P80_ASSIGNMENT_2_LAYER_H

#include <blt/std/types.h>
#include <assign2/initializers.h>
#include "blt/iterator/zip.h"
#include "blt/iterator/iterator.h"

namespace assign2
{
    class neuron_t
    {
            friend layer_t;
        public:
            // empty neuron for loading from a stream
            explicit neuron_t(weight_view weights): weights(weights)
            {}
            
            // neuron with bias
            explicit neuron_t(weight_view weights, Scalar bias): bias(bias), weights(weights)
            {}
            
            Scalar activate(const Scalar* inputs, function_t* act_func)
            {
                z = bias;
                for (auto [x, w] : blt::zip_iterator_container({inputs, inputs + weights.size()}, {weights.begin(), weights.end()}))
                    z += x * w;
                a = act_func->call(z);
                return a;
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
            float error = 0;
            weight_view weights;
    };
    
    class layer_t
    {
        public:
            template<typename WeightFunc, typename BiasFunc>
            layer_t(const blt::i32 in, const blt::i32 out, function_t* act_func, WeightFunc w, BiasFunc b):
                    in_size(in), out_size(out), act_func(act_func)
            {
                neurons.reserve(out_size);
                for (blt::i32 i = 0; i < out_size; i++)
                {
                    auto weight = weights.allocate_view(in_size);
                    for (auto& v : weight)
                        v = w(i);
                    neurons.push_back(neuron_t{weight, b(i)});
                }
            }
            
            std::vector<Scalar> call(const std::vector<Scalar>& in)
            {
                std::vector<Scalar> out;
                out.reserve(out_size);
#if BLT_DEBUG_LEVEL > 0
                if (in.size() != in_size)
                    throw std::runtime_exception("Input vector doesn't match expected input size!");
#endif
                for (auto& n : neurons)
                    out.push_back(n.activate(in.data(), act_func));
                return out;
            }
            
            Scalar back_prop(const std::vector<Scalar>& prev_layer_output, Scalar error, const layer_t& next_layer, bool is_output)
            {
                std::vector<Scalar> dw;
                
                // δ(h)
                if (is_output)
                {
                    // assign error to output layer
                    for (auto& n : neurons)
                        n.error = act_func->derivative(n.z) * error; // f'act(net(h)) * (error)
                } else
                {
                    // first calculate and assign input layer error
                    std::vector<Scalar> next_error;
                    next_error.resize(next_layer.neurons.size());
                    for (const auto& [i, w] : blt::enumerate(next_layer.neurons))
                    {
                        for (auto wv : w.weights)
                            next_error[i] += w.error * wv;
                        // needed?
                        next_error[i] /= static_cast<Scalar>(w.weights.size());
                    }
                    
                    for (auto& n : neurons)
                    {
                        n.error = act_func->derivative(n.z);
                    }
                }
                
                for (const auto& v : prev_layer_output)
                {
                
                }
                
                return error_at_current_layer;
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
        
        private:
            const blt::i32 in_size, out_size;
            weight_t weights;
            function_t* act_func;
            std::vector<neuron_t> neurons;
    };
}

#endif //COSC_4P80_ASSIGNMENT_2_LAYER_H
