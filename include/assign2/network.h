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

#ifndef COSC_4P80_ASSIGNMENT_2_NETWORK_H
#define COSC_4P80_ASSIGNMENT_2_NETWORK_H

#include <assign2/common.h>
#include <assign2/layer.h>

namespace assign2
{
    class network_t
    {
        public:
            template<typename WeightFunc, typename BiasFunc>
            network_t(blt::i32 input_size, blt::i32 output_size, blt::i32 layer_count, blt::i32 hidden_size, WeightFunc w, BiasFunc b):
                    input_size(input_size), output_size(output_size), hidden_count(layer_count), hidden_size(hidden_size)
            {
                if (layer_count > 0)
                {
                    for (blt::i32 i = 0; i < layer_count; i++)
                    {
                        if (i == 0)
                            layers.push_back(layer_t{input_size, hidden_size, w, b});
                        else
                            layers.push_back(layer_t{hidden_size, hidden_size, w, b});
                    }
                    layers.push_back(layer_t{hidden_size, output_size, w, b});
                } else
                {
                    layers.push_back(layer_t{input_size, output_size, w, b});
                }
            }
            
            template<typename WeightFunc, typename BiasFunc, typename OutputWeightFunc, typename OutputBiasFunc>
            network_t(blt::i32 input_size, blt::i32 output_size, blt::i32 layer_count, blt::i32 hidden_size,
                      WeightFunc w, BiasFunc b, OutputWeightFunc ow, OutputBiasFunc ob):
                    input_size(input_size), output_size(output_size), hidden_count(layer_count), hidden_size(hidden_size)
            {
                if (layer_count > 0)
                {
                    for (blt::i32 i = 0; i < layer_count; i++)
                    {
                        if (i == 0)
                            layers.push_back(layer_t{input_size, hidden_size, w, b});
                        else
                            layers.push_back(layer_t{hidden_size, hidden_size, w, b});
                    }
                    layers.push_back(layer_t{hidden_size, output_size, ow, ob});
                } else
                {
                    layers.push_back(layer_t{input_size, output_size, ow, ob});
                }
            }
            
            explicit network_t(std::vector<layer_t> layers):
                    input_size(layers.begin()->get_in_size()), output_size(layers.end()->get_out_size()),
                    hidden_count(static_cast<blt::i32>(layers.size()) - 1), hidden_size(layers.end()->get_in_size()), layers(std::move(layers))
            {}
            
            network_t() = default;
            
            template<typename ActFunc, typename ActFuncOut>
            std::vector<Scalar> execute(const std::vector<Scalar>& input, ActFunc func, ActFuncOut outFunc)
            {
                std::vector<Scalar> previous_output;
                std::vector<Scalar> current_output;
                
                for (auto [i, v] : blt::enumerate(layers))
                {
                    previous_output = current_output;
                    if (i == 0)
                        current_output = v.call(input, func);
                    else if (i == layers.size() - 1)
                        current_output = v.call(previous_output, outFunc);
                    else
                        current_output = v.call(previous_output, func);
                }
                
                return current_output;
            }
            
            Scalar train(const data_file_t& example)
            {
                const Scalar learn_rate = 0.1;
                
                Scalar total_error = 0;
                for (const auto& x : example.data_points)
                {
                    auto o = execute(x.bins, sigmoid_function{}, sigmoid_function{});
                    auto y = x.is_bad ? 1.0f : 0.0f;
                    
                    Scalar is_bad = 0;
                    if (o[0] >= 1)
                        is_bad = 0;
                    else if (o[1] >= 1)
                        is_bad = 1;
                    
                    auto error = y - is_bad;
                    if (o[0] >= 1 && o[1] >= 1)
                        error += 1;
                    
                    total_error += error;
                    
                }
                return total_error;
            }
        
        private:
            blt::i32 input_size, output_size, hidden_count, hidden_size;
            std::vector<layer_t> layers;
    };
}

#endif //COSC_4P80_ASSIGNMENT_2_NETWORK_H
