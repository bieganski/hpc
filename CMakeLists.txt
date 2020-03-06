cmake_minimum_required(VERSION 3.15)
project(hpc)

set(CMAKE_CXX_STANDARD 14)



include_directories(lab1)

add_executable(hpc
        lab1/hello_world.cpp
        lab1/mat_mult.cpp)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(lab1/hello_world.cpp PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(lab1/mat_mult.cpp PUBLIC OpenMP::OpenMP_CXX)
endif()