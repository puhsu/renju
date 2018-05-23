#include <chrono>
#include <string>
#include <signal.h>
#include <unistd.h>

#include "rules.h"
#include "util.h"
#include "mcts.h"

const std::string project_dir = "/Users/irubachev/Documents/cs/project/renju/src/cpp_test/";

int main(int argc, char *argv[]) {
    // load models
    Session *policy_session;
    Session *rollout_session;
    load_model(project_dir + "models/model03tf.pb", &policy_session);
    load_model(project_dir + "models/model.policy.04.pb", &rollout_session);

    // Start the game with backend
    MCTS tree(std::chrono::milliseconds{3000}, 15, policy_session, rollout_session);
    while (!exit_requested) {
        Game state = wait_for_game_update();
        tree.update_state(state);
        pos_t pos = tree.get_pos();
        send_pos_to_backend(pos.first, pos.second);
    }
    return 0;
}
