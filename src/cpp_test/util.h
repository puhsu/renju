//
//  Functions to work with tensorflow and communicate with backend enviroment
//

#ifndef GOMOKU_UTIL_H
#define GOMOKU_UTIL_H

#include <string>
#include <utility>
#include <iostream>

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"

#include "rules.h"

using tensorflow::Session;
using tensorflow::Status;
using tensorflow::Tensor;

/*
 * Create new tensorflow session and load saved graph given
 * by path to the protobuf model file
 * */
void load_model(std::string filepath, Session **session) {
    // Initialize session
    Status status = tensorflow::NewSession(tensorflow::SessionOptions(), session);
    if (!status.ok()) {
        std::cerr << status.ToString() << "\n";
        exit(1);
    }

    // Read in the protobuf graph we exported
    tensorflow::GraphDef graph_def;
    status = tensorflow::ReadBinaryProto(tensorflow::Env::Default(), filepath, &graph_def);
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


/*
 * Run session on input tensor and return probabilities
 * */
EigenTensor run_model(Session *session, Tensor input) {
    Status status;
    std::vector<std::pair<tensorflow::string, Tensor>> feed_dict = {
            {"input_1", input}
    };
    std::vector<Tensor> outputs;

    // Run the session
    status = session->Run(feed_dict, {"output_node0"}, {}, &outputs);
    if (!status.ok()) {
        std::cerr << status.ToString() << "\n";
        exit(1);
    }

    return outputs[0].matrix<float>(); // eigen tensor
}


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

#endif //GOMOKU_UTIL_H
