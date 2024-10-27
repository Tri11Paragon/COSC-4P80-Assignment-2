#include <iostream>
#include <blt/fs/loader.h>
#include <blt/parse/argparse.h>
#include <assign2/common.h>
#include <filesystem>
#include "blt/iterator/enumerate.h"
#include <assign2/layer.h>
#include <assign2/functions.h>
#include <assign2/network.h>
#include <memory>
#include <thread>

using namespace assign2;

std::vector<data_file_t> data_files;
random_init randomizer{619};
empty_init empty;
small_init small;
sigmoid_function sig;
relu_function relu;
tanh_function func_tanh;

network_t create_network(blt::i32 input, blt::i32 hidden)
{
    auto layer1 = std::make_unique<layer_t>(input, hidden * 2, &sig, randomizer, empty);
    auto layer2 = std::make_unique<layer_t>(hidden * 2, hidden / 2, &sig, randomizer, empty);
    auto layer_output = std::make_unique<layer_t>(hidden / 2, 2, &sig, randomizer, empty);
    
    std::vector<std::unique_ptr<layer_t>> vec;
    vec.push_back(std::move(layer1));
    vec.push_back(std::move(layer2));
    vec.push_back(std::move(layer_output));
    
    return network_t{std::move(vec)};
}

#ifdef BLT_USE_GRAPHICS

#include <blt/gfx/window.h>
#include "blt/gfx/renderer/resource_manager.h"
#include "blt/gfx/renderer/batch_2d_renderer.h"
#include "blt/gfx/renderer/camera.h"
#include "implot.h"
#include <imgui.h>

blt::gfx::matrix_state_manager global_matrices;
blt::gfx::resource_manager resources;
blt::gfx::batch_renderer_2d renderer_2d(resources, global_matrices);
blt::gfx::first_person_camera_2d camera;

blt::hashmap_t<blt::i32, network_t> networks;
blt::hashmap_t<blt::i32, data_file_t*> file_map;
std::unique_ptr<std::thread> network_thread;
std::atomic_bool running = true;
std::atomic_bool run_exit = true;
std::atomic_int32_t run_epoch = -1;
std::atomic_uint64_t epochs = 0;
blt::i32 time_between_runs = 0;
blt::size_t correct_recall = 0;
blt::size_t wrong_recall = 0;
bool run_network = false;

void init(const blt::gfx::window_data& data)
{
    using namespace blt::gfx;

//    auto monitor = glfwGetPrimaryMonitor();
//    auto mode = glfwGetVideoMode(monitor);
//    glfwSetWindowMonitor(data.window, monitor, 0, 0, mode->width, mode->height, mode->refreshRate);
    
    global_matrices.create_internals();
    resources.load_resources();
    renderer_2d.create();
    ImPlot::CreateContext();
    
    for (auto& f : data_files)
    {
        int input = static_cast<int>(f.data_points.begin()->bins.size());
        int hidden = input * 1;
        
        BLT_INFO("Making network of size %d", input);
        networks[input] = create_network(input, hidden);
        file_map[input] = &f;
    }
    
    errors_over_time.reserve(25000);
    error_derivative_over_time.reserve(25000);
    correct_over_time.reserve(25000);
    
    network_thread = std::make_unique<std::thread>([]() {
        while (running)
        {
            if (run_epoch >= 0)
            {
                auto error = networks.at(run_epoch).train_epoch(*file_map[run_epoch]);
                errors_over_time.push_back(error.first);
                error_derivative_over_time.push_back(error.second);
                
                blt::size_t right = 0;
                blt::size_t wrong = 0;
                for (auto& d : file_map[run_epoch]->data_points)
                {
                    auto out = networks.at(run_epoch).execute(d.bins);
                    auto is_bad = is_thinks_bad(out);
                    
                    if ((is_bad && d.is_bad) || (!is_bad && !d.is_bad))
                        right++;
                    else
                        wrong++;
                }
                correct_recall = right;
                wrong_recall = wrong;
                correct_over_time.push_back(static_cast<Scalar>(right) / static_cast<Scalar>(right + wrong) * 100);
                
                epochs++;
                run_epoch = -1;
                std::this_thread::sleep_for(std::chrono::milliseconds(time_between_runs));
            }
        }
        run_exit = false;
    });
}

template<typename Func>
void plot_vector(ImPlotRect& lims, const std::vector<Scalar>& v, std::string name, const std::string& x, const std::string& y, Func axis_func)
{
    if (lims.X.Min < 0)
        lims.X.Min = 0;
    if (ImPlot::BeginPlot(name.c_str()))
    {
        ImPlot::SetupAxes(x.c_str(), y.c_str(), ImPlotAxisFlags_None, ImPlotAxisFlags_None);
        int minX = static_cast<blt::i32>(lims.X.Min);
        int maxX = static_cast<blt::i32>(lims.X.Max);
        
        if (minX < 0)
            minX = 0;
        if (minX >= static_cast<blt::i32>(v.size()))
            minX = static_cast<blt::i32>(v.size()) - 1;
        if (maxX < 0)
            maxX = 0;
        if (maxX >= static_cast<blt::i32>(v.size()))
            maxX = static_cast<blt::i32>(v.size()) - 1;
        if (static_cast<blt::i32>(v.size()) > 0)
        {
            auto min = v[minX];
            auto max = v[minX];
            for (int i = minX; i < maxX; i++)
            {
                auto val = v[i];
                if (val < min)
                    min = val;
                if (val > max)
                    max = val;
            }
            ImPlot::SetupAxisLimits(ImAxis_Y1, axis_func(min, true), axis_func(max, false), ImGuiCond_Always);
        }
        
        name = "##" + name;
        ImPlot::SetupAxisLinks(ImAxis_X1, &lims.X.Min, &lims.X.Max);
        ImPlot::PlotLine(name.c_str(), v.data(), static_cast<int>(v.size()), 1, 0, ImPlotLineFlags_Shaded);
        ImPlot::EndPlot();
    }
}

void update(const blt::gfx::window_data& data)
{
    global_matrices.update_perspectives(data.width, data.height, 90, 0.1, 2000);
    
    camera.update();
    camera.update_view(global_matrices);
    global_matrices.update();
    
    ImGui::ShowDemoWindow();
    ImPlot::ShowDemoWindow();
    
    auto net = networks.begin();
    if (ImGui::Begin("Control", nullptr))
    {
        static std::vector<std::unique_ptr<const char>> owner;
        static std::vector<const char*> lists;
        if (lists.empty())
        {
            for (auto& n : networks)
            {
                auto str = std::to_string(n.first);
                char* ptr = new char[str.size() + 1];
                owner.push_back(std::unique_ptr<const char>(ptr));
                std::memcpy(ptr, str.data(), str.size());
                ptr[str.size()] = '\0';
                lists.push_back(ptr);
            }
        }
        static int selected = 1;
        for (int i = 0; i < selected; i++)
            net++;
        ImGui::Separator();
        ImGui::Text("Select Network Size");
        if (ImGui::ListBox("", &selected, lists.data(), static_cast<int>(lists.size()), 4))
        {
            errors_over_time.clear();
            correct_over_time.clear();
            error_derivative_over_time.clear();
            run_network = false;
        }
        ImGui::Separator();
        ImGui::Text("Using network %d size %d", selected, net->first);
        static bool pause = pause_mode.load();
        ImGui::Checkbox("Stepped Mode", &pause);
        pause_mode = pause;
        ImGui::Checkbox("Train Network", &run_network);
        if (run_network)
            run_epoch = net->first;
        ImGui::InputInt("Time Between Runs", &time_between_runs);
        if (time_between_runs < 0)
            time_between_runs = 0;
        std::string str = std::to_string(correct_recall) + "/" + std::to_string(wrong_recall + correct_recall);
        ImGui::ProgressBar(
                (wrong_recall + correct_recall != 0) ? static_cast<float>(correct_recall) / static_cast<float>(wrong_recall + correct_recall) : 0,
                ImVec2(0, 0), str.c_str());
//        const float max_learn = 100000;
//        static float learn = max_learn;
//        ImGui::SliderFloat("Learn Rate", &learn, 1, max_learn, "", ImGuiSliderFlags_Logarithmic);
//        learn_rate = learn / (max_learn * 1000);
        ImGui::Text("Learn Rate %.9f", learn_rate);
        if (ImGui::Button("Print Current"))
        {
            BLT_INFO("Test Cases:");
            blt::size_t right = 0;
            blt::size_t wrong = 0;
            for (auto& d : file_map[net->first]->data_points)
            {
                std::cout << "Good or bad? " << (d.is_bad ? "Bad" : "Good") << " :: ";
                auto out = net->second.execute(d.bins);
                auto is_bad = is_thinks_bad(out);
                
                if ((is_bad && d.is_bad) || (!is_bad && !d.is_bad))
                    right++;
                else
                    wrong++;
                
                std::cout << "NN Thinks: " << (is_bad ? "Bad" : "Good") << " || Outs: [";
                print_vec(out) << "]" << std::endl;
            }
            BLT_INFO("NN got %ld right and %ld wrong (%%%lf)", right, wrong, static_cast<double>(right) / static_cast<double>(right + wrong) * 100);
        }
    }
    ImGui::End();
    
    if (ImGui::Begin("Stats"))
    {
        static std::vector<int> x_points;
        if (errors_over_time.size() != x_points.size())
        {
            x_points.clear();
            for (int i = 0; i < static_cast<int>(errors_over_time.size()); i++)
                x_points.push_back(i);
        }
        
        auto domain = static_cast<int>(errors_over_time.size());
        blt::i32 history = std::min(100, domain);
        static ImPlotRect lims(0, 100, 0, 1);
        if (ImPlot::BeginAlignedPlots("AlignedGroup"))
        {
            plot_vector(lims, errors_over_time, "Error", "Time", "Error", [](auto v, bool b) {
                float percent = 0.15;
                if (b)
                    return v < 0 ? v * (1 + percent) : v * (1 - percent);
                else
                    return v < 0 ? v * (1 - percent) : v * (1 + percent);
            });
            plot_vector(lims, correct_over_time, "Correct", "Time", "Correct", [](auto v, bool b) {
                if (b)
                    return v - 1;
                else
                    return v + 1;
            });
            plot_vector(lims, error_derivative_over_time, "DError/Dw", "Time", "Error", [](auto v, bool b) {
                float percent = 0.05;
                if (b)
                    return v < 0 ? v * (1 + percent) : v * (1 - percent);
                else
                    return v < 0 ? v * (1 - percent) : v * (1 + percent);
            });
            ImPlot::EndAlignedPlots();
        }
    }
    ImGui::End();
    
    
    ImGui::Begin("Hello", nullptr,
                 ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoInputs |
                 ImGuiWindowFlags_NoTitleBar);
    net->second.render();
    ImGui::End();
    
    renderer_2d.render(data.width, data.height);
}

void destroy()
{
    running = false;
    while (run_exit)
    {
        if (pause_mode)
            pause_flag = true;
    }
    if (network_thread->joinable())
        network_thread->join();
    network_thread = nullptr;
    networks.clear();
    file_map.clear();
    ImPlot::DestroyContext();
    global_matrices.cleanup();
    resources.cleanup();
    renderer_2d.cleanup();
    blt::gfx::cleanup();
}

#endif

int main(int argc, const char** argv)
{
    blt::arg_parse parser;
    parser.addArgument(blt::arg_builder("-f", "--file").setHelp("path to the data files").setDefault("../data").build());
    
    auto args = parser.parse_args(argc, argv);
    std::string data_directory = blt::string::ensure_ends_with_path_separator(args.get<std::string>("file"));
    
    data_files = load_data_files(get_data_files(data_directory));

#ifdef BLT_USE_GRAPHICS
    blt::gfx::init(blt::gfx::window_data{"Freeplay Graphics", init, update, 1440, 720}.setSyncInterval(1).setMonitor(glfwGetPrimaryMonitor())
                                                                                      .setMaximized(true));
    destroy();
    return 0;
#endif
    
    for (auto f : data_files)
    {
        int input = static_cast<int>(f.data_points.begin()->bins.size());
        int hidden = input * 3;
        
        if (input != 64)
            continue;
        
        BLT_INFO("-----------------");
        BLT_INFO("Running for size %d", input);
        BLT_INFO("With hidden layers %d", input);
        BLT_INFO("-----------------");
        
        network_t network = create_network(input, hidden);
        
        for (blt::size_t i = 0; i < 2000; i++)
            network.train_epoch(f);
        
        BLT_INFO("Test Cases:");
        blt::size_t right = 0;
        blt::size_t wrong = 0;
        for (auto& d : f.data_points)
        {
            std::cout << "Good or bad? " << (d.is_bad ? "Bad" : "Good") << " :: ";
            auto out = network.execute(d.bins);
            auto is_bad = is_thinks_bad(out);
            
            if ((is_bad && d.is_bad) || (!is_bad && !d.is_bad))
                right++;
            else
                wrong++;
            
            std::cout << "NN Thinks: " << (is_bad ? "Bad" : "Good") << " || Outs: [";
            print_vec(out) << "]" << std::endl;
        }
        BLT_INFO("NN got %ld right and %ld wrong (%%%lf)", right, wrong, static_cast<double>(right) / static_cast<double>(right + wrong) * 100);
    }
    
    std::cout << "Hello World!" << std::endl;
}
