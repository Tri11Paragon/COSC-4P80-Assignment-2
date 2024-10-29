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
//            explicit neuron_t(weight_view weights, weight_view dw): dw(dw), weights(weights)
//            {}
            
            // neuron with bias
            explicit neuron_t(weight_view weights, weight_view dw, weight_view momentum, Scalar bias):
                    bias(bias), dw(dw), weights(weights), momentum(momentum)
            {}
            
            Scalar activate(const std::vector<Scalar>& inputs, function_t* act_func)
            {
                BLT_ASSERT_MSG(inputs.size() == weights.size(), (std::to_string(inputs.size()) + " vs " + std::to_string(weights.size())).c_str());
                
                z = bias;
                for (auto [x, w] : blt::zip_iterator_container({inputs.begin(), inputs.end()}, {weights.begin(), weights.end()}))
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
            
            void update(float omega, bool reset)
            {
                // if omega is zero we are not using momentum.
                if (reset || omega == 0)
                {
//                    BLT_TRACE("Momentum Reset");
//                    for (auto& v : momentum)
//                        std::cout << v << ',';
//                    std::cout << std::endl;
                    for (auto& m : momentum)
                        m = 0;
                } else
                {
                    for (auto [m, d] : blt::in_pairs(momentum, dw))
                        m += omega * d;
                }
                for (auto [w, m, d] : blt::zip(weights, momentum, dw))
                    w += m + d;
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
            weight_view momentum;
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
                momentum.preallocate(in_size * out_size);
                for (blt::i32 i = 0; i < out_size; i++)
                {
                    auto weight = weights.allocate_view(in_size);
                    auto dw = weight_derivatives.allocate_view(in_size);
                    auto m = momentum.allocate_view(in_size);
                    for (auto& v : weight)
                        v = w(i);
                    neurons.push_back(neuron_t{weight, dw, m, b(i)});
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
                    outputs.push_back(n.activate(in, act_func));
                return outputs;
            }
            
            error_data_t back_prop(const std::vector<Scalar>& prev_layer_output,
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
//                                if (outputs[0] > 0.3 && outputs[1] > 0.3)
//                                    d *= 10 * (outputs[0] + outputs[1]);
                                auto d2 = 0.5f * (d * d);
                                // according to the slides and the 3b1b video we sum on the squared error
                                // not sure why on the slides the 1/2 is moved outside the sum as the cost function is defined (1/2) * (o - y)^2
                                // and that the total cost for an input pattern is the sum of costs on the output
                                total_error += d2;
                                total_derivative += d;
                                n.back_prop(act_func, prev_layer_output, d);
                            }
                        },
                        // interior layer
                        [this, &prev_layer_output](const layer_t& layer) {
                            for (auto [i, n] : blt::enumerate(neurons))
                            {
                                // TODO: this is not efficient on the cache!
                                Scalar w = 0;
                                for (auto nn : layer.neurons)
                                    w += nn.error * nn.weights[i];
                                n.back_prop(act_func, prev_layer_output, w);
                            }
                        }
                }, data);
                return {total_error, total_derivative};
            }
            
            void update(const float* omega, bool reset)
            {
                for (auto& n : neurons)
                    n.update(omega == nullptr ? 0 : *omega, reset);
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
            
            void render(blt::gfx::batch_renderer_2d& renderer) const
            {
                const blt::size_t distance_between_layers = 30;
                const float neuron_size = 30;
                const float padding = -5;
                for (const auto& [i, n] : blt::enumerate(neurons))
                {
                    auto color = std::abs(n.a);
                    renderer.drawPointInternal(blt::make_color(0.1, 0.1, 0.1),
                                               blt::gfx::point2d_t{static_cast<float>(i) * (neuron_size + padding) + neuron_size,
                                                                   static_cast<float>(layer_id * distance_between_layers) + neuron_size,
                                                                   neuron_size / 2}, 10);
                    auto outline_size = neuron_size + 10;
                    renderer.drawPointInternal(blt::make_color(color, color, color),
                                               blt::gfx::point2d_t{static_cast<float>(i) * (neuron_size + padding) + neuron_size,
                                                                   static_cast<float>(layer_id * distance_between_layers) + neuron_size,
                                                                   outline_size / 2}, 0);
//                    const ImVec2 alignment = ImVec2(0.5f, 0.5f);
//                    if (i > 0)
//                        ImGui::SameLine();
//                    ImGui::PushStyleVar(ImGuiStyleVar_SelectableTextAlign, alignment);
//                    std::string name;
//                    name = std::to_string(n.a);
//                    ImGui::Selectable(name.c_str(), false, ImGuiSelectableFlags_None, ImVec2(80, 80));
//                    ImGui::PopStyleVar();
                }
            }

#endif
        
        private:
            const blt::i32 in_size, out_size;
            const blt::size_t layer_id;
            weight_t weights;
            weight_t weight_derivatives;
            weight_t momentum;
            function_t* act_func;
            std::vector<neuron_t> neurons;
            std::vector<Scalar> outputs;
    };
}

#endif //COSC_4P80_ASSIGNMENT_2_LAYER_H
