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

#include <Eigen/Dense>
#include <blt/std/types.h>

namespace assign2
{
    
    class layer_t
    {
        public:
            using matrix_t = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;
            using vector_t = Eigen::Matrix<float, Eigen::Dynamic, 1>;
            
            layer_t(const blt::i32 in, const blt::i32 out): in_size(in), out_size(out)
            {}
            
            template<typename ActFunction>
            vector_t forward(const vector_t & in, ActFunction func = ActFunction{})
            {
                vector_t out;
                out.resize(out_size, Eigen::NoChange_t{});
                out.noalias() = weights.transpose() * in;
                out.colwise() += bias;
                return func(out);
            }
        
        private:
            const blt::i32 in_size, out_size;
            matrix_t weights{};
            vector_t bias{};
    };
}

#endif //COSC_4P80_ASSIGNMENT_2_LAYER_H
