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

#ifndef COSC_4P80_ASSIGNMENT_2_INITIALIZERS_H
#define COSC_4P80_ASSIGNMENT_2_INITIALIZERS_H

#include <blt/std/types.h>
#include <blt/std/random.h>
#include <assign2/common.h>

namespace assign2
{
    struct empty_init
    {
        template<int rows, int columns>
        inline void operator()(Eigen::Matrix<Scalar, rows, columns>& matrix) const
        {
            for (auto r : matrix.rowwise())
            {
                for (auto& v : r)
                    v = 0;
            }
        }
    };
    
    struct half_init
    {
        template<int rows, int columns>
        inline void operator()(Eigen::Matrix<Scalar, rows, columns>& matrix) const
        {
            for (auto r : matrix.rowwise())
            {
                for (auto& v : r)
                    v = 0.5f;
            }
        }
    };
    
    struct random_init
    {
        public:
            explicit random_init(blt::size_t seed, float min = 0.5 - 0.125, float max = 0.5 + 0.125): seed(seed), min(min), max(max)
            {}
            
            template<int rows, int columns>
            inline void operator()(Eigen::Matrix<Scalar, rows, columns>& matrix) const
            {
                blt::random::random_t random(seed);
                for (auto r : matrix.rowwise())
                {
                    for (auto& v : r)
                        v = random.get_float(min, max);
                }
            }
        
        private:
            blt::size_t seed;
            float min, max;
    };
    
}

#endif //COSC_4P80_ASSIGNMENT_2_INITIALIZERS_H
