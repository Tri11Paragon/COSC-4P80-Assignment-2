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

#ifndef COSC_4P80_ASSIGNMENT_2_COMMON_H
#define COSC_4P80_ASSIGNMENT_2_COMMON_H

#include <Eigen/Dense>

namespace assign2
{
    struct data_t
    {
        bool is_bad = false;
        std::vector<float> bins;
    };
    
    struct data_file_t
    {
        std::vector<data_t> data_points;
    };
    
    class layer_t;
    class network_t;
    
    using Scalar = float;
    using matrix_t = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using vector_t = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
}

#endif //COSC_4P80_ASSIGNMENT_2_COMMON_H
