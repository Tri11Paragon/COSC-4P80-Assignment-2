// Copyright (c) 2024. Brett Terpstra
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

#include <assign2/file.h>
#include <blt/std/string.h>
#include <blt/fs/loader.h>
#include <filesystem>
#include <cmath>
#include <fstream>

namespace assign2
{
    std::vector<std::string> data_file_t::get_data_file_list(std::string_view path)
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
    
    std::vector<data_file_t> data_file_t::load_data_files(const std::vector<std::string>& files)
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
                
                for (++line_data_it; line_data_it != line_data_meta.end(); ++line_data_it)
                    line_data.bins.push_back(std::stof(*line_data_it));
                
                data.data_points.push_back(line_data);
            }
            
            loaded_data.push_back(data);
        }
        
        return loaded_data;
    }
    
    std::vector<data_file_t> data_file_t::load_data_files_from_path(std::string_view path)
    {
        return load_data_files(get_data_file_list(path));
    }
    
    data_t data_t::with_padding(blt::size_t desired_size, Scalar padding_value) const
    {
        data_t data = *this;
        auto amount_to_add = static_cast<blt::ptrdiff_t>(data.bins.size()) - static_cast<blt::ptrdiff_t>(desired_size);
        for (blt::ptrdiff_t i = 0; i < amount_to_add; i++)
            data.bins.push_back(padding_value);
        return data;
    }
    
    data_t data_t::normalize() const
    {
        data_t data = *this;
        
        Scalar total = 0;
        for (auto v : data.bins)
            total += v * v;
        Scalar mag = std::sqrt(total);
        for (auto& v : data.bins)
            v /= mag;
        return data;
    }
    
    data_file_t data_file_t::normalize() const
    {
        auto copy = *this;
        
        for (auto& v : copy.data_points)
            v = v.normalize();
        
        return copy;
    }
    
    data_file_t data_file_t::with_padding(blt::size_t desired_size, Scalar padding_value) const
    {
        auto copy = *this;
        
        for (auto& v : copy.data_points)
            v = v.with_padding(desired_size, padding_value);
        
        return copy;
    }
    
    data_file_t& data_file_t::operator+=(const data_file_t& o)
    {
        data_points.insert(data_points.end(), o.data_points.begin(), o.data_points.end());
        return *this;
    }
    
    data_file_t operator+(const data_file_t& a, const data_file_t& b)
    {
        data_file_t file = a;
        file.data_points.insert(file.data_points.end(), b.data_points.begin(), b.data_points.end());
        return file;
    }
    
    void save_as_csv(const std::string& file, const std::vector<std::pair<std::string, std::vector<Scalar>>>& data)
    {
        std::ofstream stream{file};
        stream << "epoch,";
        for (auto [i, d] : blt::enumerate(data))
        {
            stream << d.first;
            if (i != data.size() - 1)
                stream << ',';
        }
        stream << '\n';
        for (blt::size_t i = 0; i < data.begin()->second.size(); i++)
        {
            stream << i << ',';
            for (auto [j, d] : blt::enumerate(data))
            {
                stream << d.second[i];
                if (j != data.size() - 1)
                    stream << ',';
            }
            stream << '\n';
        }
    }
}