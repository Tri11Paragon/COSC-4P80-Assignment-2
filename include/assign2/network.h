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
#include "global_magic.h"

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
                            layers.push_back(std::make_unique<layer_t>(input_size, hidden_size, w, b));
                        else
                            layers.push_back(std::make_unique<layer_t>(hidden_size, hidden_size, w, b));
                    }
                    layers.push_back(std::make_unique<layer_t>(hidden_size, output_size, w, b));
                } else
                {
                    layers.push_back(std::make_unique<layer_t>(input_size, output_size, w, b));
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
                            layers.push_back(std::make_unique<layer_t>(input_size, hidden_size, w, b));
                        else
                            layers.push_back(std::make_unique<layer_t>(hidden_size, hidden_size, w, b));
                    }
                    layers.push_back(std::make_unique<layer_t>(hidden_size, output_size, ow, ob));
                } else
                {
                    layers.push_back(std::make_unique<layer_t>(input_size, output_size, ow, ob));
                }
            }
            
            explicit network_t(std::vector<std::unique_ptr<layer_t>> layers): layers(std::move(layers))
            {}
            
            network_t() = default;
            
            const std::vector<Scalar>& execute(const std::vector<Scalar>& input)
            {
                std::vector<blt::ref<const std::vector<Scalar>>> outputs;
                outputs.emplace_back(input);
                
                for (auto [i, v] : blt::enumerate(layers))
                    outputs.emplace_back(v->call(outputs.back()));
                
                return outputs.back();
            }
            
            error_data_t error(const data_file_t& data)
            {
                Scalar total_error = 0;
                Scalar total_d_error = 0;
                
                for (auto& d : data.data_points)
                {
                    std::vector<Scalar> expected{d.is_bad ? 0.0f : 1.0f, d.is_bad ? 1.0f : 0.0f};
                    
                    auto out = execute(d.bins);
                    
                    BLT_ASSERT(out.size() == expected.size());
                    for (auto [o, e] : blt::in_pairs(out, expected))
                    {
                        auto d_error = e - o;
                        
                        auto error = 0.5f * (d_error * d_error);
                        
                        total_error += error;
                        total_d_error += d_error;
                    }
                }
                
                return {total_error / static_cast<Scalar>(data.data_points.size()), total_d_error / static_cast<Scalar>(data.data_points.size())};
            }
            
            error_data_t train(const data_t& data, bool reset)
            {
                error_data_t error = {0, 0};
                execute(data.bins);
                std::vector<Scalar> expected{data.is_bad ? 0.0f : 1.0f, data.is_bad ? 1.0f : 0.0f};
                
                for (auto [i, layer] : blt::iterate(layers).enumerate().rev())
                {
                    if (i == layers.size() - 1)
                    {
                        error += layer->back_prop(layers[i - 1]->outputs, expected);
                    } else if (i == 0)
                    {
                        error += layer->back_prop(data.bins, *layers[i + 1]);
                    } else
                    {
                        error += layer->back_prop(layers[i - 1]->outputs, *layers[i + 1]);
                    }
                }
                for (auto& l : layers)
                    l->update(m_omega, reset);
//                BLT_TRACE("Error for input: %f, derr: %f", error.error, error.d_error);
                return error;
            }
            
            error_data_t train_epoch(const data_file_t& example, blt::i32 trains_per_data = 1)
            {
                error_data_t error{0, 0};
                for (const auto& x : example.data_points)
                {
                    for (blt::i32 i = 0; i < trains_per_data; i++)
                        error += train(x, reset_next);
                }
                // take the average cost over all the training.
                error.d_error /= static_cast<Scalar>(example.data_points.size() * trains_per_data);
                error.error /= static_cast<Scalar>(example.data_points.size() * trains_per_data);
                // as long as we are reducing error in the same direction in overall terms, we should still build momentum.
                auto last_sign = last_d_error >= 0;
                auto cur_sign = error.d_error >= 0;
                last_d_error = error.d_error;
                reset_next = last_sign != cur_sign;
                return error;
            }
            
            void with_momentum(Scalar* omega)
            {
                m_omega = omega;
            }

#ifdef BLT_USE_GRAPHICS
            
            void render(blt::gfx::batch_renderer_2d& renderer) const
            {
                for (auto& l : layers)
                    l->render(renderer);
            }

#endif
        
        private:
            // pointer so it can be changed from the UI
            Scalar* m_omega = nullptr;
            Scalar last_d_error = 0;
            bool reset_next = false;
            std::vector<std::unique_ptr<layer_t>> layers;
    };
}

#endif //COSC_4P80_ASSIGNMENT_2_NETWORK_H
