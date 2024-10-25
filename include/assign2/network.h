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
            network_t(blt::i32 input_size, blt::i32 output_size, blt::i32 layer_count, blt::i32 hidden_size, WeightFunc w, BiasFunc b)
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
                      WeightFunc w, BiasFunc b, OutputWeightFunc ow, OutputBiasFunc ob)
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
            
            explicit network_t(std::vector<layer_t> layers): layers(std::move(layers))
            {}
            
            network_t() = default;
            
            const std::vector<Scalar>& execute(const std::vector<Scalar>& input)
            {
                std::vector<blt::ref<const std::vector<Scalar>>> outputs;
                outputs.emplace_back(input);
                
                for (auto& v : layers)
                    outputs.emplace_back(v.call(outputs.back()));
                
                return outputs.back();
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
            
            Scalar train_epoch(const data_file_t& example)
            {
                Scalar total_error = 0;
                Scalar total_d_error = 0;
                for (const auto& x : example.data_points)
                {
                    execute(x.bins);
                    std::vector<Scalar> expected{x.is_bad ? 0.0f : 1.0f, x.is_bad ? 1.0f : 0.0f};
                    
                    for (auto [i, layer] : blt::iterate(layers).enumerate().rev())
                    {
                        if (i == layers.size() - 1)
                        {
                            auto e = layer.back_prop(layers[i - 1].outputs, expected);
                            total_error += e;
                        } else if (i == 0)
                        {
                            auto e = layer.back_prop(x.bins, layers[i + 1]);
                            total_error += e;
                        } else
                        {
                            auto e = layer.back_prop(layers[i - 1].outputs, layers[i + 1]);
                            total_error += e;
                        }
                    }
                    for (auto& l : layers)
                        l.update();
                }
                BLT_DEBUG("Total Errors found %f, %f", total_error, total_d_error);
                
                return total_error;
            }
        
        private:
            std::vector<layer_t> layers;
    };
}

#endif //COSC_4P80_ASSIGNMENT_2_NETWORK_H
