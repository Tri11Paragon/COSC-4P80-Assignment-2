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

#ifndef COSC_4P80_ASSIGNMENT_2_GLOBAL_MAGIC_H
#define COSC_4P80_ASSIGNMENT_2_GLOBAL_MAGIC_H

#include <vector>
#include <unordered_map>
#include <assign2/common.h>
#include <blt/math/vectors.h>
#include <atomic>
#include <thread>

namespace assign2
{
    
    inline blt::size_t layer_id_counter = 0;
    inline std::atomic_bool pause_mode = true;
    inline std::atomic_bool pause_flag = false;
    
    void await()
    {
        if (!pause_mode.load(std::memory_order_relaxed))
            return;
        // wait for flag to come in
        while (!pause_flag.load(std::memory_order_relaxed))
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        // reset the flag back to false
        auto flag = pause_flag.load(std::memory_order_relaxed);
        while (!pause_flag.compare_exchange_strong(flag, false, std::memory_order_relaxed))
        {}
    }
    
    struct node_data
    {
    
    };
    
    inline std::vector<Scalar> errors_over_time;
    inline std::vector<Scalar> error_derivative_over_time;
    inline std::vector<Scalar> error_of_test;
    inline std::vector<Scalar> error_of_test_derivative;
    
    inline std::vector<Scalar> error_derivative_of_test;
    inline std::vector<Scalar> correct_over_time;
    inline std::vector<Scalar> correct_over_time_test;
    inline std::vector<node_data> nodes;
    
    void save_error_info(const std::string& name)
    {
        save_as_csv("network" + name + ".csv", {{"train_error",   errors_over_time},
                                                                        {"train_d_error", error_derivative_over_time},
                                                                        {"test_error",    error_of_test},
                                                                        {"test_d_error",  error_of_test_derivative},
                                                                        {"correct_train",       correct_over_time},
                                                                        {"correct_test",       correct_over_time_test}});
    }
}

#endif //COSC_4P80_ASSIGNMENT_2_GLOBAL_MAGIC_H
