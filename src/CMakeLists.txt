cmake_minimum_required(VERSION 3.9)
project(bernard)

set(CMAKE_CXX_STANDARD 11)

include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/libs/tensorflow/include
        ${CMAKE_CURRENT_SOURCE_DIR}/libs/google/include
        ${CMAKE_CURRENT_SOURCE_DIR}/libs/tensorflow/include/external/nsync/public)

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/libs/tensorflow/lib/osx/)
add_executable(bernard main.cpp rules.hpp util.hpp mcts.hpp policy.hpp rollout.hpp)
target_link_libraries(bernard tensorflow_cc protobuf)
