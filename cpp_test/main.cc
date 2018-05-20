#include <iostream>
#include <limits>
#include <chrono>

#include "rules.h"

int main(int argc, char *argv[]) {
    Session *session;
    load_model("../models/model.policy.04.pb", &session);

    Game game;
    game.get_board().setRandom();

    
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    for (int i = 0; i < 2342; ++i) {
        auto prob = run_model(session, game.get_state());
        // std::cout << std::endl << prob << std::endl;
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Running time (sec) = " 
              << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000000.0 
              << std::endl;

    // Free any resources used by the session
    Status status = session->Close();
    if (!status.ok()) {
        std::cout << status.ToString() << "\n";
        exit(1);
    }

    return 0;
}
