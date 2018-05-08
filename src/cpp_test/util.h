#pragma once
#include <string>
#include <unordered_map>
#include <cstdlib>

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/cc/framework/ops.h"

using tensorflow::Session;
using tensorflow::Status;
using tensorflow::Tensor;


typedef Eigen::Array<float, 15, 15, Eigen::RowMajor> EigenArray;
typedef Eigen::Tensor<float, 2, Eigen::RowMajor> EigenTensor;
typedef std::pair<int, int> pos_t;


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
    return POS_TO_LETTER[pos.first] + std::to_string(pos.second);
}

pos_t to_pos(std::string move) {
    if (LETTER_TO_POS.size() != POS_TO_LETTER.size()) {
        for (int i = 0; i < POS_TO_LETTER.length(); ++i) {
            LETTER_TO_POS[POS_TO_LETTER[i]] = i;
        }
    }
    return {move[1] - '0', LETTER_TO_POS[move[0]]};
}





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

    // Free any resources used by the session
    status = session->Close();
    if (!status.ok()) {
        std::cout << status.ToString() << "\n";
        exit(1);
    }

    return outputs[0].matrix<float>();
}
