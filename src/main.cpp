#include <chrono>
#include <string>
#include <signal.h>
#include <unistd.h>

#include "rules.hpp"
#include "util.hpp"
#include "mcts.hpp"
// load models 
#include "rollout.hpp"
#include "policy.hpp"



int main(int argc, char *argv[]) {
    // load models
    Session *policy_session;
    Session *rollout_session;
    load_model(policy_pb, policy_pb_len, &policy_session);
    std::cerr << "[Bernard] Loaded policy model" << std::endl;
    load_model(rollout_pb, rollout_pb_len, &rollout_session);
    std::cerr << "[Bernard] Loaded rollout model" << std::endl;

    // Start game with backend
    MCTS tree(std::chrono::milliseconds{3000}, 10, policy_session, rollout_session);
    std::cerr << "[Bernard] Initialized MCTS class" << std::endl;
    while (true) {
        created_count = 0;
        deleted_count = 0;
        Game state = wait_for_game_update();
        if (!state) {
            break;
        }
        tree.update_state(state);

        auto beg = std::chrono::high_resolution_clock::now();
        pos_t pos = tree.get_pos();
        auto end = std::chrono::high_resolution_clock::now();
        std::cerr << "[Bernard] Created nodes " << created_count << std::endl;
        std::cerr << "[Bernard] Deleted nodes " << deleted_count << std::endl;
        std::chrono::duration<double> elapsed = end - beg;
        std::cerr << "[Bernard] Sent pos (" << pos.first << ", " << pos.second << ")\n";
        std::cerr << "[Bernard] Time elapsed " << elapsed.count() << std::endl;
        send_pos_to_backend(pos.first, pos.second);
    }
    return 0;
}
