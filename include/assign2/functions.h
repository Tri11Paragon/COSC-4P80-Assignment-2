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

#ifndef COSC_4P80_ASSIGNMENT_2_FUNCTIONS_H
#define COSC_4P80_ASSIGNMENT_2_FUNCTIONS_H

#include <assign2/common.h>
#include <cmath>

namespace assign2
{
    struct sigmoid_function
    {
        [[nodiscard]] Scalar call(Scalar s) const // NOLINT
        {
            return 1 / (1 + std::exp(-s));
        }
        
        [[nodiscard]] Scalar derivative(Scalar s) const
        {
            return call(s) * (1 - call(s));
        }
    };
    
    struct linear_function
    {
    
    };
}

#endif //COSC_4P80_ASSIGNMENT_2_FUNCTIONS_H
