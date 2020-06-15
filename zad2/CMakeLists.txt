cmake_minimum_required(VERSION 3.10)
project(body)
set(CMAKE_CXX_STANDARD 14)

set(SOURCE_FILES src/main.cpp src/utils.cpp src/body.cpp)

add_executable(body3 ${SOURCE_FILES})

find_package(MPI)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
target_link_libraries(body3 ${MPI_C_LIBRARIES})
