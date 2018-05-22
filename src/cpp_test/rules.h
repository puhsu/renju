#pragma once
#include <iostream>
#include <string>
#include <memory>
#include <cassert>

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/cc/framework/ops.h"

#include "logger.h"

using tensorflow::Session;
using tensorflow::Status;
using tensorflow::Tensor;

typedef Eigen::Array<float, 15, 15, Eigen::RowMajor> EigenArray;
typedef Eigen::Tensor<float, 2, Eigen::RowMajor> EigenTensor;
typedef std::pair<int, int> pos_t;


ige::FileLogger lg ("1.0", "logfile.txt");

// GAME UTIL FUNCTIONS
enum Color {
    NONE = 0,
    BLACK = -1,
    WHITE = 1
};

Color another(Color c) {
    if (c == WHITE) {
        return BLACK;
    }

    if (c == BLACK) {
        return WHITE;
    }

    return NONE;
}


std::string POS_TO_LETTER = "abcdefghjklmnop";
std::unordered_map<char, int> LETTER_TO_POS;

std::string to_move(pos_t pos) {
    return POS_TO_LETTER[pos.second] + std::to_string(pos.first + 1);
}

pos_t to_pos(std::string move) {
    if (LETTER_TO_POS.size() != POS_TO_LETTER.size()) {
        for (int i = 0; i < POS_TO_LETTER.length(); ++i) {
            LETTER_TO_POS[POS_TO_LETTER[i]] = i;
        }
    }
    return {stoi(move.substr(1, move.length())) - 1, LETTER_TO_POS[move[0]]};
}


// TENSORFLOW UTIL FUNCTIONS

// create new session and load saved graph
void load_model(std::string modelfile, Session **session) {
    // Initialize session
    Status status = tensorflow::NewSession(tensorflow::SessionOptions(), session);
    if (!status.ok()) {
        std::cerr << status.ToString() << "\n";
        exit(1);
    }

    // Read in the protobuf graph we exported
    tensorflow::GraphDef graph_def;
    status = tensorflow::ReadBinaryProto(tensorflow::Env::Default(), modelfile, &graph_def);
    if (!status.ok()) {
        std::cerr << status.ToString() << "\n";
        exit(1);
    }

    // Add the graph to the session
    status = (*session)->Create(graph_def);
    if (!status.ok()) {
        std::cerr << status.ToString() << "\n";
        exit(1);
    }
}


// run model on input tensor and return matrix of probabilities
EigenTensor run_model(Session *session, Tensor input) {
    Status status;
    std::vector<std::pair<tensorflow::string, Tensor>> feed_dict = {
        {"input_1", input}
    };
    std::vector<Tensor> outputs;

    // Run the session
    status = session->Run(feed_dict, {"output_node0"}, {}, &outputs);
    if (!status.ok()) {
        std::cout << status.ToString() << "\n";
        exit(1);
    }

    return outputs[0].matrix<float>();
}




class Game {
private:
    static constexpr int width = 15;
    static constexpr int height = 15;
    static const int line_length = 5;

    Color result = NONE;
    Color player = BLACK;

    EigenArray board;
    std::vector<std::pair<int, int>> positions;


    // Set of functions to check if game terminates after each move
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

        return check_row(i, j)
            || check_col(i, j)
            || check_main_diag(i, j)
            || check_side_diag(i, j);
    }

public:
    Game() {
        // default constructor
        board.setZero();
        positions.resize(0);
    }

    Game(std::string data) {
        // construct game from existing game data with moves
        // used to communicate with python gui and other agents

        board.setZero();
        positions.resize(0);

        std::vector<std::string> moves;
        std::istringstream iss(data);
        std::copy(std::istream_iterator<std::string>(iss),
                  std::istream_iterator<std::string>(),
                  std::back_inserter(moves));

        for (auto m : moves) {
            pos_t pos = to_pos(m);
            move(pos.first, pos.second);
        }
    }

    // Copy constructor : deepcopy
    Game(const Game& other) {
        result = other.result;
        player = other.player;
        board = other.board;
        positions = other.positions;
    }


    Game& operator=(Game other) {
        // Assignment operator: does a deep copy
        result = other.result;
        player = other.player;
        board = other.board;
        positions = other.positions;
        return *this;
    }

    
    bool is_possible_move(int i, int j) const {
        bool row = (0 <= i) && (i < height);
        bool col = (0 <= j) && (j < width);
        return row && col && !board(i, j);
    }


    void move(int i, int j) {
        if (!is_possible_move(i, j)) {
            lg << i << j;
        }
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
        // Construct a tensorflow Tensor object, which represents 
        // current game state input to policy network

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
        return state;
    }


    size_t move_n() const {
        return positions.size();
    }


    pos_t last_pos() const {
        return positions[positions.size() - 1];
    }


    explicit operator bool() const {
        return get_result() == NONE && move_n() < width * height;
    }


    void print_board() const {
        std::cout << board << std::endl;
    }
};


namespace backend {
    Game wait_for_game_update() {
        std::string data;
        std::getline(std::cin, data);
        //std::cout << "got string: " << data << std::endl;
        return Game(data);
    }
    
    void send_pos_to_backend(int i, int j) {
        std::cout << to_move({i, j}) << "\n";
        std::cout.flush();
    }
};

