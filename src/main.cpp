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
#include <algorithm>
#include <mutex>

using namespace assign2;

std::vector<data_file_t> data_files;
blt::hashmap_t<blt::i32, std::vector<data_file_t>> groups;
blt::hashmap_t<blt::i32, network_t> networks;
bool with_momentum = false;
Scalar omega = 0.001;

random_init randomizer{std::random_device{}()};
empty_init empty;
small_init small;
sigmoid_function sig;
relu_function relu;
bulu_function bulu;
tanh_function func_tanh;

network_t create_network(blt::i32 input, blt::i32 hidden)
{
    const auto mul = 0.5;
    const auto inner_mul = 0.25;
    auto layer1 = std::make_unique<layer_t>(input, hidden * mul, &sig, randomizer, empty);
    auto layer2 = std::make_unique<layer_t>(hidden * mul, hidden * inner_mul, &sig, randomizer, empty);
//    auto layer3 = std::make_unique<layer_t>(hidden * inner_mul, hidden * inner_mul, &sig, randomizer, empty);
//    auto layer4 = std::make_unique<layer_t>(hidden * inner_mul, hidden * inner_mul, &sig, randomizer, empty);
    auto layer_output = std::make_unique<layer_t>(hidden * inner_mul, 2, &sig, randomizer, empty);
    
    std::vector<std::unique_ptr<layer_t>> vec;
    vec.push_back(std::move(layer1));
    vec.push_back(std::move(layer2));
//    vec.push_back(std::move(layer3));
//    vec.push_back(std::move(layer4));
    vec.push_back(std::move(layer_output));
    
    network_t network{std::move(vec)};
    if (with_momentum)
        network.with_momentum(&omega);
    return network;
}

std::pair<data_file_t, data_file_t> create_groups(blt::i32 network, blt::i32 k = 0)
{
    data_file_t training;
    data_file_t testing;
    
    testing.data_points.insert(testing.data_points.begin(),
                               (groups[network].begin() + k)->data_points.begin(),
                               (groups[network].begin() + k)->data_points.end());
    
    for (auto [i, a] : blt::enumerate(groups[network]))
    {
        if (i == static_cast<blt::size_t>(k))
            continue;
        training.data_points.insert(training.data_points.begin(), a.data_points.begin(), a.data_points.end());
    }
    
    return {training, testing};
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

data_file_t current_training;
data_file_t current_testing;
std::atomic_int32_t run_epoch = -1;
blt::i32 stop_at = -1;
blt::i32 trains_per_data = 1;
std::mutex vec_lock;

std::unique_ptr<std::thread> network_thread;
std::atomic_bool running = true;
std::atomic_bool run_exit = true;
std::atomic_uint64_t epochs = 0;
blt::i32 time_between_runs = 0;
blt::i32 number_before_switch = 10;
bool swap_k_after = false;
blt::size_t correct_recall_train = 0;
blt::size_t correct_recall_test = 0;
blt::size_t wrong_recall_train = 0;
blt::size_t wrong_recall_test = 0;
bool run_network = false;

float init_learn = learn_rate;
float init_momentum = omega;

blt::i32 current_k = 0;

void update_current(int network)
{
    if (groups[network].size() > 1)
    {
        std::scoped_lock lock(vec_lock);
        current_testing.data_points.clear();
        current_training.data_points.clear();
        
        auto g = create_groups(network, current_k);
        current_testing = g.second;
        current_training = g.first;
    } else
    {
        std::scoped_lock lock(vec_lock);
        current_training = groups[network].front();
        current_testing = groups[network].front();
    }
}

void reset_errors(int network)
{
    save_error_info(std::to_string(network));
    errors_over_time.clear();
    correct_over_time.clear();
    correct_over_time_test.clear();
    error_derivative_over_time.clear();
    error_of_test.clear();
    error_of_test_derivative.clear();
    epochs = 0;
    run_network = false;
}

void init(const blt::gfx::window_data&)
{
    using namespace blt::gfx;

//    auto monitor = glfwGetPrimaryMonitor();
//    auto mode = glfwGetVideoMode(monitor);
//    glfwSetWindowMonitor(data.window, monitor, 0, 0, mode->width, mode->height, mode->refreshRate);
    
    global_matrices.create_internals();
    resources.load_resources();
    renderer_2d.create();
    ImPlot::CreateContext();
    
    update_current(networks.begin()->first);
    
    network_thread = std::make_unique<std::thread>([]() {
        while (running)
        {
            if (run_epoch >= 0)
            {
                if (swap_k_after && epochs % number_before_switch == static_cast<blt::size_t>(number_before_switch - 1))
                {
                    current_k++;
                    current_k %= static_cast<blt::i32>(groups[run_epoch].size());
                    update_current(run_epoch);
                }
                
                blt::size_t right_t = 0;
                blt::size_t wrong_t = 0;
                blt::size_t right_a = 0;
                blt::size_t wrong_a = 0;
                {
                    std::scoped_lock lock(vec_lock);
                    auto error = networks.at(run_epoch).train_epoch(current_training, trains_per_data);
                    errors_over_time.push_back(error.error);
                    error_derivative_over_time.push_back(error.d_error);
                    
                    auto error_test = networks.at(run_epoch).error(current_testing);
                    error_of_test.push_back(error_test.error);
                    error_of_test_derivative.push_back(error_test.d_error);
                    
                    for (auto& d : current_testing.data_points)
                    {
                        auto out = networks.at(run_epoch).execute(d.bins);
                        auto is_bad = is_thinks_bad(out);
                        
                        if ((is_bad && d.is_bad) || (!is_bad && !d.is_bad))
                            right_t++;
                        else
                            wrong_t++;
                    }
                    
                    for (auto& d : current_training.data_points)
                    {
                        auto out = networks.at(run_epoch).execute(d.bins);
                        auto is_bad = is_thinks_bad(out);
                        
                        if ((is_bad && d.is_bad) || (!is_bad && !d.is_bad))
                            right_a++;
                        else
                            wrong_a++;
                    }
                }
                correct_recall_test = right_t;
                correct_recall_train = right_a;
                wrong_recall_test = wrong_t;
                wrong_recall_train = wrong_a;
                correct_over_time
                        .push_back(static_cast<Scalar>(correct_recall_train) / static_cast<Scalar>(correct_recall_train + wrong_recall_train) * 100);
                correct_over_time_test
                        .push_back(static_cast<Scalar>(correct_recall_test) / static_cast<Scalar>(correct_recall_test + wrong_recall_test) * 100);
                
                auto error = errors_over_time.back();
//                error = std::sqrt(error * error + error + 0.01f);
//                error = std::max(0.0f, std::min(1.0f, error));
                learn_rate = error * init_learn;
                omega = error * init_momentum;
                
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

static void HelpMarker(const char* desc)
{
    ImGui::TextDisabled("(?)");
    if (ImGui::BeginItemTooltip())
    {
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
        ImGui::TextUnformatted(desc);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
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
        static int selected = 0;
        for (int i = 0; i < selected; i++)
            net++;
        ImGui::Separator();
        ImGui::Text("Select Network Size");
        if (ImGui::ListBox("", &selected, lists.data(), static_cast<int>(lists.size()), 4))
        {
            reset_errors(net->first);
            net = networks.begin();
            for (int i = 0; i < selected; i++)
                net++;
            update_current(net->first);
        }
        ImGui::Separator();
        ImGui::Text("Using network %d size %d", selected, net->first);
        ImGui::Checkbox("Train Network", &run_network);
        ImGui::InputInt("Stop At", &stop_at);
        if (static_cast<blt::i32>(epochs) >= stop_at && stop_at > 0)
            run_network = false;
        if (run_network)
        {
//            update_current(net->first);
            run_epoch = net->first;
        }
        ImGui::InputInt("Time Between Runs", &time_between_runs);
        if (time_between_runs < 0)
            time_between_runs = 0;
        std::string str = std::to_string(correct_recall_test) + "/" + std::to_string(wrong_recall_test + correct_recall_test);
        ImGui::ProgressBar(
                (wrong_recall_test + correct_recall_test != 0) ? static_cast<float>(correct_recall_test) /
                                                                 static_cast<float>(wrong_recall_test + correct_recall_test) : 0,
                ImVec2(0, 0), str.c_str());
        ImGui::Separator();
        str = std::to_string(correct_recall_train) + "/" + std::to_string(wrong_recall_train + correct_recall_train);
        ImGui::ProgressBar(
                (wrong_recall_train + correct_recall_train != 0) ? static_cast<float>(correct_recall_train) /
                                                                   static_cast<float>(wrong_recall_train + correct_recall_train) : 0,
                ImVec2(0, 0), str.c_str());
        ImGui::Text("Learn Rate %.9f", learn_rate);
        if (ImGui::Button("Print Current"))
        {
            BLT_INFO("Test Cases:");
            blt::size_t right = 0;
            blt::size_t wrong = 0;
            for (auto& d : current_testing.data_points)
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
        if (ImGui::SliderInt("K For Testing", &current_k, 0, static_cast<int>(groups[net->first].size() - 1)))
            update_current(net->first);
        ImGui::Checkbox("Auto-swap K", &swap_k_after);
        if (swap_k_after)
        {
            ImGui::InputInt("Number of epochs before switch", &number_before_switch);
            if (number_before_switch < 1)
                number_before_switch = 1;
        }
        ImGui::Checkbox("Momentum", &with_momentum);
        ImGui::SameLine();
        HelpMarker("You might want to reset the network after changing this");
        if (with_momentum)
            ImGui::SliderFloat("##MomentumSlider", &omega, 0, 0.1, "%.8f", ImGuiSliderFlags_Logarithmic);
        ImGui::InputInt("Trains per Epoch", &trains_per_data);
        ImGui::SameLine();
        HelpMarker("Number of times to run back-prop on a piece of data before moving on to the next");
        if (trains_per_data < 1)
            trains_per_data = 1;
        ImGui::Separator();
        if (ImGui::Button("Reset Network"))
        {
            reset_errors(net->first);
            layer_id_counter = 0;
            networks[net->first] = create_network(net->first, net->first);
        }
        ImGui::Separator();
        if (ImGui::Button("Save current to CSV"))
            save_error_info(std::to_string(net->first) + "_" + std::to_string(current_k));
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
        
        static ImPlotRect lims(0, 500, 0, 1);
        if (ImPlot::BeginSubplots("##LinkedGroup", 3, 2, ImVec2(-1, -1)))
        {
            plot_vector(lims, errors_over_time, "Global Error (Training)", "Epoch", "Error", [](auto v, bool b) {
                float percent = 0.15;
                if (b)
                    return v < 0 ? v * (1 + percent) : v * (1 - percent);
                else
                    return v < 0 ? v * (1 - percent) : v * (1 + percent);
            });
            plot_vector(lims, error_of_test, "Global Error (Tests)", "Epoch", "Error", [](auto v, bool b) {
                float percent = 0.15;
                if (b)
                    return v < 0 ? v * (1 + percent) : v * (1 - percent);
                else
                    return v < 0 ? v * (1 - percent) : v * (1 + percent);
            });
            plot_vector(lims, error_derivative_over_time, "DError/Dw (Training)", "Epoch", "DError", [](auto v, bool b) {
                float percent = 0.05;
                if (b)
                    return v < 0 ? v * (1 + percent) : v * (1 - percent);
                else
                    return v < 0 ? v * (1 - percent) : v * (1 + percent);
            });
            plot_vector(lims, error_of_test_derivative, "DError/Dw (Test)", "Epoch", "DError", [](auto v, bool b) {
                float percent = 0.05;
                if (b)
                    return v < 0 ? v * (1 + percent) : v * (1 - percent);
                else
                    return v < 0 ? v * (1 - percent) : v * (1 + percent);
            });
            plot_vector(lims, correct_over_time, "% Correct (Training)", "Epoch", "Correct%", [](auto v, bool b) {
                if (b)
                    return v - 1;
                else
                    return v + 1;
            });
            plot_vector(lims, correct_over_time_test, "% Correct (Test)", "Epoch", "Correct%", [](auto v, bool b) {
                if (b)
                    return v - 1;
                else
                    return v + 1;
            });
            ImPlot::EndSubplots();
        }
    }
    ImGui::End();
    
    
    ImGui::Begin("Hello", nullptr,
                 ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoInputs |
                 ImGuiWindowFlags_NoTitleBar);
    net->second.render(renderer_2d);
    ImGui::End();
    
    renderer_2d.render(data.width, data.height);
}

void destroy()
{
    save_error_info(std::to_string(run_epoch));
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
    parser.addArgument(blt::arg_builder("-f", "--file").setHelp("Path to the data files").setDefault("../data").setMetavar("FOLDER").build());
    parser.addArgument(
            blt::arg_builder("-k", "--kfold").setHelp("Number of groups to split into [Defaults to 3 if no number is provided]")
                                             .setAction(blt::arg_action_t::STORE).setNArgs('?').setConst("3").setMetavar("GROUPS").build());
    parser.addArgument(blt::arg_builder("-m", "--momentum").setHelp("Use momentum in weight calculations").setAction(blt::arg_action_t::STORE_TRUE)
                                                           .setDefault(false).build());
    
    auto args = parser.parse_args(argc, argv);
    if (args.get<bool>("momentum"))
    {
        BLT_INFO("Using Momentum");
        with_momentum = true;
    }
    
    std::string data_directory = blt::string::ensure_ends_with_path_separator(args.get<std::string>("file"));
    
    data_files = load_data_files(get_data_files(data_directory));
    
    if (args.contains("kfold"))
    {
        auto kfold = std::stoul(args.get<std::string>("kfold"));
        BLT_INFO("Running K-Fold-%ld", kfold);
        blt::random::random_t rand(std::random_device{}());
        for (auto& n : data_files)
        {
            std::vector<data_t> goods;
            // Big Airship of Doom (BAD)
            std::vector<data_t> bads;
            
            for (auto& p : n.data_points)
            {
                if (p.is_bad)
                    bads.push_back(p);
                else
                    goods.push_back(p);
            }
            
            // can randomize the order of good and bad inputs
            std::shuffle(goods.begin(), goods.end(), rand);
            std::shuffle(bads.begin(), bads.end(), rand);
            
            auto size = static_cast<blt::i32>(n.data_points.begin()->bins.size());
            groups[size] = {};
            for (blt::size_t i = 0; i < kfold; i++)
                groups[size].emplace_back();
            
            // then copy proportionally into the groups, creating roughly equal groups of data.
            // my previous setup randomly selected the group index
            // this resulted in wildly uneven groups, if you got unlucky.
            // 25 vs 13 in some groups
            // not sure if this is what we want, but it felt like this would create issues
            blt::size_t select = 0;
            for (auto& v : goods)
            {
                ++select %= kfold;
                groups[size][select].data_points.push_back(v);
            }
            
            // because bad motors are in a separate step they are still proportional
            for (auto& v : bads)
            {
                ++select %= kfold;
                groups[size][select].data_points.push_back(v);
            }
        }
    } else
    {
        for (auto& n : data_files)
        {
            auto size = static_cast<blt::i32>(n.data_points.begin()->bins.size());
            groups[size].push_back(n);
        }
    }
    
    for (const auto& [set, g] : groups)
    {
        BLT_INFO("Set %d has groups %ld", set, g.size());
        for (auto [i, f] : blt::enumerate(g))
            BLT_INFO("\tData file %ld contains %ld elements", i + 1, f.data_points.size());
    }
    
    for (auto& f : data_files)
    {
        int input = static_cast<int>(f.data_points.begin()->bins.size());
        int hidden = input * 1;
        
        BLT_INFO("Making network of size %d", input);
        layer_id_counter = 0;
        networks[input] = create_network(input, hidden);
    }
    
    // this is to prevent threading issues due to expanding buffers.
    errors_over_time.reserve(25000);
    error_derivative_over_time.reserve(25000);
    correct_over_time.reserve(25000);
    correct_over_time_test.reserve(25000);
    error_of_test.reserve(25000);
    error_of_test_derivative.reserve(25000);

#ifdef BLT_USE_GRAPHICS
    blt::gfx::init(blt::gfx::window_data{"Freeplay Graphics", init, update, 1440, 720}.setSyncInterval(1).setMonitor(glfwGetPrimaryMonitor())
                                                                                      .setMaximized(true));
    destroy();
    return 0;
#endif
    
    for (auto f : data_files)
    {
        int input = static_cast<int>(f.data_points.begin()->bins.size());
        int hidden = input;
        
        if (input != 64)
            continue;
        
        BLT_INFO("-----------------");
        BLT_INFO("Running for size %d", input);
        BLT_INFO("With hidden layers %d", input);
        BLT_INFO("-----------------");
        
        network_t network = create_network(input, hidden);
        
        float o = 0.00001;
//        network.with_momentum(&o);
        for (blt::size_t i = 0; i < 10000; i++)
            network.train_epoch(f, 1);
        
        BLT_INFO("Test Cases:");
        blt::size_t right = 0;
        blt::size_t wrong = 0;
        for (auto& d : f.data_points)
        {
            auto out = network.execute(d.bins);
            auto is_bad = is_thinks_bad(out);
            
            if ((is_bad && d.is_bad) || (!is_bad && !d.is_bad))
                right++;
            else
                wrong++;
            
            std::cout << "Good or bad? " << (d.is_bad ? "Bad" : "Good") << " :: ";
            std::cout << "NN Thinks: " << (is_bad ? "Bad" : "Good") << " || Outs: [";
            print_vec(out) << "]" << std::endl;
        }
        BLT_INFO("NN got %ld right and %ld wrong (%%%lf)", right, wrong, static_cast<double>(right) / static_cast<double>(right + wrong) * 100);
    }
    
    std::cout << "Hello World!" << std::endl;
}
