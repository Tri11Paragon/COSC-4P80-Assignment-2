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
#include <filesystem>

#ifdef BLT_USE_GRAPHICS
    
    #include "blt/gfx/renderer/batch_2d_renderer.h"
    #include "blt/gfx/window.h"
    #include <imgui.h>

#endif

namespace assign2
{
    using Scalar = float;
//    const inline Scalar learn_rate = 0.001;
    inline Scalar learn_rate = 0.001;
    
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
            weight_t() = default;
            
            weight_t(const weight_t& copy) = delete;
            
            weight_t& operator=(const weight_t& copy) = delete;
            
            weight_t(weight_t&& move) noexcept: place(std::exchange(move.place, 0)), data(std::move(move.data))
            {}
            
            weight_t& operator=(weight_t&& move) noexcept
            {
                place = std::exchange(move.place, place);
                data = std::exchange(move.data, std::move(data));
                return *this;
            }
            
            void preallocate(blt::size_t amount)
            {
                data.resize(amount);
            }
            
            weight_view allocate_view(blt::size_t count)
            {
                auto size = place;
                place += count;
                return {&data[size], count};
            }
            
            void debug() const
            {
                std::cout << "Weights: ";
                print_vec(data) << std::endl;
            }
        
        private:
            blt::size_t place = 0;
            std::vector<Scalar> data;
    };
    
    std::vector<std::string> get_data_files(std::string_view path)
    {
        std::vector<std::string> files;
        
        for (const auto& file : std::filesystem::recursive_directory_iterator(path))
        {
            if (file.is_directory())
                continue;
            auto file_path = file.path().string();
            if (blt::string::ends_with(file_path, ".out"))
                files.push_back(blt::fs::getFile(file_path));
        }
        
        return files;
    }
    
    std::vector<data_file_t> load_data_files(const std::vector<std::string>& files)
    {
        std::vector<data_file_t> loaded_data;
        
        // load all file
        for (auto file : files)
        {
            // we only use unix line endings here...
            blt::string::replaceAll(file, "\r", "");
            auto lines = blt::string::split(file, "\n");
            auto line_it = lines.begin();
            auto meta = blt::string::split(*line_it, ' ');
            
            // load data inside files
            data_file_t data;
            data.data_points.reserve(std::stoll(meta[0]));
            auto bin_count = std::stoul(meta[1]);
            
            for (++line_it; line_it != lines.end(); ++line_it)
            {
                auto line_data_meta = blt::string::split(*line_it, ' ');
                if (line_data_meta.size() != bin_count + 1)
                    continue;
                auto line_data_it = line_data_meta.begin();
                
                // load bins
                data_t line_data;
                line_data.is_bad = std::stoi(*line_data_it) == 1;
                line_data.bins.reserve(bin_count);
                Scalar total = 0;
                for (++line_data_it; line_data_it != line_data_meta.end(); ++line_data_it)
                {
                    auto v = std::stof(*line_data_it);
                    total += v * v;
                    line_data.bins.push_back(v);
                }
                
                // normalize vector.
                total = std::sqrt(total);
//
                for (auto& v : line_data.bins)
                    v /= total;
//
//            if (line_data.bins.size() == 32)
//                print_vec(line_data.bins) << std::endl;
                
                data.data_points.push_back(line_data);
            }
            
            loaded_data.push_back(data);
        }
        
        return loaded_data;
    }
    
    bool is_thinks_bad(const std::vector<Scalar>& out)
    {
        return out[0] < out[1];
    }
    
}

#endif //COSC_4P80_ASSIGNMENT_2_COMMON_H
