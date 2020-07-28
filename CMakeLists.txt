cmake_minimum_required(VERSION 3.10.2)

project(vk_matmul_perf)

find_package(Vulkan REQUIRED)

include_directories(${Vulkan_INCLUDE_DIR})

add_executable(vk_matmul_perf src/vk_matmul_perf.cpp)

target_link_libraries(vk_matmul_perf ${Vulkan_LIBRARY})
