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

#ifndef COSC_4P80_ASSIGNMENT_2_FILE_H
#define COSC_4P80_ASSIGNMENT_2_FILE_H

#include <assign2/common.h>


namespace assign2
{
    
    struct data_t
    {
        bool is_bad = false;
        std::vector<Scalar> bins;
        
        [[nodiscard]] data_t normalize() const;
        [[nodiscard]] data_t with_padding(blt::size_t desired_size, Scalar padding_value = 0) const;
    };
    
    struct data_file_t
    {
        public:
            std::vector<data_t> data_points;
            
            static std::vector<data_file_t> load_data_files_from_path(std::string_view path);
        
        private:
            static std::vector<std::string> get_data_file_list(std::string_view path);
            
            static std::vector<data_file_t> load_data_files(const std::vector<std::string>& files);
    };
    
    void save_as_csv(const std::string& file, const std::vector<std::pair<std::string, std::vector<Scalar>>>& data);
    
}

#endif //COSC_4P80_ASSIGNMENT_2_FILE_H
