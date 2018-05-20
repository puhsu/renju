cmake_minimum_required(VERSION 3.9)
project(gomoku)

set(CMAKE_CXX_STANDARD 11)

include_directories(
        ${CMAKE_SOURCE_DIR}/libs/tensorflow/include
        ${CMAKE_SOURCE_DIR}/libs/google/include
        ${CMAKE_SOURCE_DIR}/libs/tensorflow/include/external/nsync/public)


link_directories(${CMAKE_SOURCE_DIR}/libs/tensorflow/lib/osx/)
add_executable(gomoku main.cc rules.h)
target_link_libraries(gomoku tensorflow_cc)
