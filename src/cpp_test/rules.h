#pragma once
#include <iostream>
#include <string>
#include <memory>
#include <cassert>


#include "util.h"


class Game {
private:
    static constexpr int width = 15;
    static constexpr int height = 15;
    const int line_length = 5;

    Color result = NONE;
    Color player = BLACK;

    EigenArray board;
    std::vector<std::pair<int, int>> positions;

    
    bool check_row(int i, int j) const {
        int length = 1;
        int y = 1;
        while (j-y >= 0 && board(i, j-y) == board(i, j)) {
            ++length;
            ++y;
        }
        y = 1;
        while (j+y < width && board(i, j+y) == board(i, j)) {
            ++length;
            ++y;
        }
        return (length >= line_length);
    }

    bool check_col(int i, int j) const {
        int length = 1;
        int x = 1;
        while (i-x >= 0 && board(i-x, j) == board(i, j)) {
            ++length;
            ++x;
        }
        x = 1;
        while (i+x < height && board(i+x, j) == board(i, j)) {
            ++length;
            ++x;
        }
        return (length >= line_length);        
    }

    bool check_main_diag(int i, int j) const {
        //check for main diagnol
        int length = 1;
        int x = 1;
        while (i-x >= 0 && j-x >= 0 && board(i-x, j-x) == board(i, j)) {
            ++length;
            ++x;
        }
        x = 1;
        while (i+x < height && j+x < width && board(i+x, j+x) == board(i, j)) {
            ++length;
            ++x;
        }
        return (length >= line_length);
    }

    bool check_side_diag(int i, int j) const {
        //check for main diagnol
        int length = 1;
        int x = 1;
        while (i-x >= 0 && j+x < width && board(i-x, j+x) == board(i, j)) {
            ++length;
            ++x;
        }
        x = 1;
        while (i+x < height && j-x >= 0 && board(i+x, j-x) == board(i, j)) {
            ++length;
            ++x;
        }
        return (length >= line_length);        
    }

    bool check_pos(int i, int j) const {
        if (!board(i, j)) {
            return false;
        }

        return check_row(i, j) \
            || check_col(i, j) \
            || check_main_diag(i, j) \
            || check_side_diag(i, j);
    }

public:
    Game() {
        board.setZero();
        positions.resize(0);
    }
    
    bool is_possible_move(int i, int j) const {
        bool row = 0 <= i && i < height;
        bool col = 0 <= j && j < width;
        return row && col && !board(i, j);
    }


    void move(int i, int j) {
        assert(is_possible_move(i, j));
        
        positions.push_back({i, j});
        board(i, j) = player;

        if (!result && check_pos(i, j)) {
            result = player;
            return;
        }

        player = another(player);
    }


    Color get_result() const {
        return result;
    }

    Color get_player() const {
        return player;
    }

    EigenArray& get_board() {
        return board;
    }

    Tensor get_state() const {
        Tensor state = Tensor(tensorflow::DT_FLOAT, {1, 15, 15, 4});
        auto tensor_map = state.tensor<float, 4>();
        tensor_map.setZero();

        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                if (board(i, j) == player) {
                    tensor_map(0, i, j, 0) = 1;
                }
                if (board(i, j) == another(player)) {
                    tensor_map(0, i, j, 1) = 1;
                }
                if (player == BLACK) {
                    tensor_map(0, i, j, 2) = 1;
                }
            }
        }

        // std::cout << tensor_map.chip(0, 3).reshape(Eigen::array<long,2>{15,15}) << std::endl;
        // std::cout << tensor_map.chip(1, 3).reshape(Eigen::array<long,2>{15,15}) << std::endl;
        // std::cout << tensor_map.chip(2, 3).reshape(Eigen::array<long,2>{15,15}) << std::endl;
        // std::cout << tensor_map.chip(3, 3).reshape(Eigen::array<long,2>{15,15}) << std::endl;

        return state;
    }

    int move_n() const {
        return positions.size();
    }

    pos_t last_pos() const {
        return positions[positions.size() - 1];
    }

    operator bool() const {
        return get_result() == NONE && move_n() < width * height;
    }

    void print_board() const {
        std::cout << board << std::endl;
    }
};
