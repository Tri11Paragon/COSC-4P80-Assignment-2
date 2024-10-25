#include <iostream>
#include <blt/fs/loader.h>
#include <blt/parse/argparse.h>
#include <assign2/common.h>
#include <filesystem>
#include "blt/iterator/enumerate.h"
#include <assign2/layer.h>
#include <assign2/functions.h>
#include <assign2/network.h>

using namespace assign2;

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
            for (++line_data_it; line_data_it != line_data_meta.end(); ++line_data_it)
            {
                line_data.bins.push_back(std::stof(*line_data_it));
            }
            data.data_points.push_back(line_data);
        }
        
        loaded_data.push_back(data);
    }
    
    return loaded_data;
}

int main(int argc, const char** argv)
{
    blt::arg_parse parser;
    parser.addArgument(blt::arg_builder("-f", "--file").setHelp("path to the data files").setDefault("../data").build());
    
    auto args = parser.parse_args(argc, argv);
    std::string data_directory = blt::string::ensure_ends_with_path_separator(args.get<std::string>("file"));
    
    auto data_files = load_data_files(get_data_files(data_directory));
    
    random_init randomizer{619};
    empty_init empty;
    sigmoid_function sig;
    relu_function relu;
    threshold_function thresh;
    
    layer_t layer1{16, 16, &sig, randomizer, empty};
    layer1.debug();
    layer_t layer2{16, 16, &sig, randomizer, empty};
    layer2.debug();
    layer_t layer3{16, 16, &sig, randomizer, empty};
    layer3.debug();
    layer_t layer_output{16, 2, &sig, randomizer, empty};
    layer_output.debug();
    
    network_t network{{layer1, layer2, layer3, layer_output}};
    
    for (auto f : data_files)
    {
        if (f.data_points.begin()->bins.size() == 16)
        {
            for (blt::size_t i = 0; i < 10; i++)
            {
                network.train_epoch(f);
            }
            break;
        }
    }
    BLT_INFO("Test Cases:");
    
    for (auto f : data_files)
    {
        if (f.data_points.begin()->bins.size() == 16)
        {
            for (auto& d : f.data_points)
            {
                std::cout << "Good or bad? " << d.is_bad << " :: ";
                print_vec(network.execute(d.bins)) << std::endl;
            }
        }
    }
//    for (auto d : data_files)
//    {
//        BLT_TRACE_STREAM << "\nSilly new file:\n";
//        for (auto point : d.data_points)
//        {
//            BLT_TRACE_STREAM << "Is bad? " << (point.is_bad ? "True" : "False") << " [";
//            for (auto [i, bin] : blt::enumerate(point.bins))
//            {
//                BLT_TRACE_STREAM << bin;
//                if (i != point.bins.size()-1)
//                    BLT_TRACE_STREAM << ", ";
//            }
//            BLT_TRACE_STREAM << "]\n";
//        }
//    }
    
    
    
    std::cout << "Hello World!" << std::endl;
}
