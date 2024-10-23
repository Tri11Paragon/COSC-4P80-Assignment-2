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
        inline Scalar operator()(blt::i32) const
        {
            return 0;
        }
    };
    
    struct half_init
    {
        inline Scalar operator()(blt::i32) const
        {
            return 0;
        }
    };
    
    struct random_init
    {
        public:
            explicit random_init(blt::size_t seed, Scalar min = -0.5, Scalar max = 0.5): random(seed), seed(seed), min(min), max(max)
            {}
            
            inline Scalar operator()(blt::i32)
            {
                return static_cast<Scalar>(random.get_double(min, max));
            }
        
        private:
            blt::random::random_t random;
            blt::size_t seed;
            Scalar min, max;
    };
    
}

#endif //COSC_4P80_ASSIGNMENT_2_INITIALIZERS_H
