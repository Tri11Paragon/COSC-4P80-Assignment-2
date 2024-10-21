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
            network_t(blt::i32 input_size, blt::i32 output_size, blt::i32 hidden_count, blt::i32 hidden_size):
                    input_size(input_size), output_size(output_size), hidden_count(hidden_count), hidden_size(hidden_size)
            {
                if (hidden_count > 0)
                {
                    layers.push_back(layer_t{input_size, hidden_size});
                    for (blt::i32 i = 1; i < hidden_count; i++)
                        layers.push_back(layer_t{hidden_size, hidden_size});
                    layers.push_back(layer_t{hidden_size, output_size});
                } else
                    layers.push_back(layer_t{input_size, output_size});
            }
        
        private:
            blt::i32 input_size, output_size, hidden_count, hidden_size;
            std::vector<layer_t> layers;
    };
}

#endif //COSC_4P80_ASSIGNMENT_2_NETWORK_H
