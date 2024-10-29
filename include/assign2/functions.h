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
    struct sigmoid_function : public function_t
    {
        [[nodiscard]] Scalar call(const Scalar s) const final
        {
            return 1 / (1 + std::exp(-s));
        }
        
        [[nodiscard]] Scalar derivative(const Scalar s) const final
        {
            auto v = call(s);
            return v * (1 - v);
        }
    };
    
    struct tanh_function : public function_t
    {
        [[nodiscard]] Scalar call(Scalar s) const final
        {
            auto x = std::exp(s);
            auto nx = std::exp(-s);
            return (x - nx) / (x + nx);
        }
        
        [[nodiscard]] Scalar derivative(Scalar s) const final
        {
            auto tanh = std::tanh(s);
            return 1 - (tanh * tanh);
        }
    };
    
    struct relu_function : public function_t
    {
        [[nodiscard]] Scalar call(const Scalar s) const final
        {
            return std::max(static_cast<Scalar>(0), s);
        }
        
        [[nodiscard]] Scalar derivative(Scalar s) const final
        {
            return s >= 0 ? 1 : 0;
        }
    };
    
    struct bulu_function : public function_t
    {
        [[nodiscard]] Scalar call(const Scalar s) const final
        {
            return s > 0.5 ? s : -s;
        }
        
        [[nodiscard]] Scalar derivative(Scalar s) const final
        {
            return s >= 0 ? 1 : -1;
        }
    };
}

#endif //COSC_4P80_ASSIGNMENT_2_FUNCTIONS_H
