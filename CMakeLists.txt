cmake_minimum_required(VERSION 3.25)
project(COSC-4P80-Assignment-2 VERSION 0.1.1)

option(ENABLE_ADDRSAN "Enable the address sanitizer" OFF)
option(ENABLE_UBSAN "Enable the ub sanitizer" OFF)
option(ENABLE_TSAN "Enable the thread data race sanitizer" OFF)
option(ENABLE_GRAPHICS "Enable usage of graphics package" OFF)
#option(EIGEN_TEST_CXX11 "Enable testing with C++11 and C++11 features (e.g. Tensor module)." ON)

set(CMAKE_CXX_STANDARD 17)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

if (ENABLE_GRAPHICS)
    add_subdirectory(lib/blt-graphics)
    add_compile_definitions(BLT_USE_GRAPHICS)
    set(EXTRA_SOURCES lib/implot/implot.cpp lib/implot/implot_demo.cpp lib/implot/implot_items.cpp)
    include_directories(lib/implot)
else ()
    add_subdirectory(lib/blt)
endif ()

#add_subdirectory(lib/eigen-3.4.0)

include_directories(include/)
file(GLOB_RECURSE PROJECT_BUILD_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_executable(COSC-4P80-Assignment-2 ${PROJECT_BUILD_FILES} ${EXTRA_SOURCES})

target_compile_options(COSC-4P80-Assignment-2 PRIVATE -Wall -Wextra -Wpedantic -Wno-comment)
target_link_options(COSC-4P80-Assignment-2 PRIVATE -Wall -Wextra -Wpedantic -Wno-comment)

#target_link_libraries(COSC-4P80-Assignment-2 PRIVATE BLT Eigen3::Eigen)
if (ENABLE_GRAPHICS)
    target_link_libraries(COSC-4P80-Assignment-2 PRIVATE BLT_WITH_GRAPHICS)
else ()
    target_link_libraries(COSC-4P80-Assignment-2 PRIVATE BLT)
endif ()

if (${ENABLE_ADDRSAN} MATCHES ON)
    target_compile_options(COSC-4P80-Assignment-2 PRIVATE -fsanitize=address)
    target_link_options(COSC-4P80-Assignment-2 PRIVATE -fsanitize=address)
endif ()

if (${ENABLE_UBSAN} MATCHES ON)
    target_compile_options(COSC-4P80-Assignment-2 PRIVATE -fsanitize=undefined)
    target_link_options(COSC-4P80-Assignment-2 PRIVATE -fsanitize=undefined)
endif ()

if (${ENABLE_TSAN} MATCHES ON)
    target_compile_options(COSC-4P80-Assignment-2 PRIVATE -fsanitize=thread)
    target_link_options(COSC-4P80-Assignment-2 PRIVATE -fsanitize=thread)
endif ()
