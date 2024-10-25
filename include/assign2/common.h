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

#include <iostream>
#include <blt/iterator/enumerate.h>

namespace assign2
{
    using Scalar = float;
    const inline Scalar learn_rate = 0.1;
    
    template<typename T>
    decltype(std::cout)& print_vec(const std::vector<T>& vec)
    {
        for (auto [i, v] : blt::enumerate(vec))
        {
            std::cout << v;
            if (i != vec.size() - 1)
                std::cout << ", ";
        }
        return std::cout;
    }
    
    struct data_t
    {
        bool is_bad = false;
        std::vector<Scalar> bins;
    };
    
    struct data_file_t
    {
        std::vector<data_t> data_points;
    };
    
    class layer_t;
    
    class network_t;
    
    struct function_t
    {
        [[nodiscard]] virtual Scalar call(Scalar) const = 0;
        
        [[nodiscard]] virtual Scalar derivative(Scalar) const = 0;
    };
    
    struct weight_view
    {
        public:
            weight_view(Scalar* data, blt::size_t size): m_data(data), m_size(size)
            {}
            
            inline Scalar& operator[](blt::size_t index) const
            {
#if BLT_DEBUG_LEVEL > 0
                if (index >= size)
                    throw std::runtime_error("Index is out of bounds!");
#endif
                return m_data[index];
            }
            
            [[nodiscard]] inline blt::size_t size() const
            {
                return m_size;
            }
            
            [[nodiscard]] auto begin() const
            {
                return m_data;
            }
            
            [[nodiscard]] auto end() const
            {
                return m_data + m_size;
            }
        
        private:
            Scalar* m_data;
            blt::size_t m_size;
    };
    
    /**
     * this class exists purely as an optimization
     */
    class weight_t
    {
        public:
            weight_view allocate_view(blt::size_t count)
            {
                auto size = data.size();
                data.resize(size + count);
                return {&data[size], count};
            }
            
            void debug() const
            {
                std::cout << "Weights: ";
                print_vec(data) << std::endl;
            }
        
        private:
            std::vector<Scalar> data;
    };
    
}

#endif //COSC_4P80_ASSIGNMENT_2_COMMON_H
