cmake_minimum_required(VERSION 3.9)
project(gomoku)

set(CMAKE_CXX_STANDARD 11)

include_directories(
        ~/Documents/cs/project/renju/src/cpp_test/libs/tensorflow/include
        ~/Documents/cs/project/renju/src/cpp_test/libs/google/include
        ~/Documents/cs/project/renju/src/cpp_test/libs/tensorflow/include/external/nsync/public)


link_directories(~/Documents/cs/project/renju/src/cpp_test/libs/tensorflow/lib/osx/)
add_executable(gomoku main.cc rules.h util.h mcts.h policy.h rollout.h)
target_link_libraries(gomoku tensorflow_cc protobuf)
