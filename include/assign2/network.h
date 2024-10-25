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
#include "blt/std/assert.h"

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
            
            std::vector<Scalar> execute(const std::vector<Scalar>& input)
            {
                std::vector<Scalar> previous_output;
                std::vector<Scalar> current_output;
                
                for (auto [i, v] : blt::enumerate(layers))
                {
                    previous_output = current_output;
                    if (i == 0)
                        current_output = v.call(input);
                    else
                        current_output = v.call(previous_output);
                }
                
                return current_output;
            }
            
            std::pair<Scalar, Scalar> error(const std::vector<Scalar>& outputs, bool is_bad)
            {
                BLT_ASSERT(outputs.size() == 2);
                auto g = is_bad ? 0.0f : 1.0f;
                auto b = is_bad ? 1.0f : 0.0f;
                
                auto g_diff = outputs[0] - g;
                auto b_diff = outputs[1] - b;
                
                auto error = g_diff * g_diff + b_diff * b_diff;
                BLT_INFO("%f %f %f", error, g_diff, b_diff);
                
                return {0.5f * (error * error), error};
            }
            
            Scalar train(const data_file_t& example)
            {
                Scalar total_error = 0;
                Scalar total_d_error = 0;
                for (const auto& x : example.data_points)
                {
                    print_vec(x.bins) << std::endl;
                    auto o = execute(x.bins);
                    print_vec(o) << std::endl;
                    auto [e, de] = error(o, x.is_bad);
                    total_error += e;
                    total_d_error += -learn_rate * de;
                    BLT_TRACE("\tError %f, %f, is bad? %s", e, -learn_rate * de, x.is_bad ? "True" : "False");
                }
                BLT_DEBUG("Total Errors found %f, %f", total_error, total_d_error);
                
                return total_error;
            }
        
        private:
            blt::i32 input_size, output_size, hidden_count, hidden_size;
            std::vector<layer_t> layers;
    };
}

#endif //COSC_4P80_ASSIGNMENT_2_NETWORK_H
