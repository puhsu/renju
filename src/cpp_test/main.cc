#include <chrono>
#include <string>
#include <signal.h>
#include <unistd.h>

#include "rules.h"
#include "util.h"
#include "mcts.h"
// weights for models
#include "rollout.h"
#include "policy.h"

int main(int argc, char *argv[]) {
    // load models
    Session *policy_session;
    Session *rollout_session;
    load_model(model_policy, model_policy_len, &policy_session);
    load_model(model_rollout, model_rollout_len, &rollout_session);

    // Start the game with backend
    MCTS tree(std::chrono::milliseconds{1000}, 15, policy_session, rollout_session);
    while (true) {
        Game state = wait_for_game_update();
        tree.update_state(state);
        pos_t pos = tree.get_pos();
        send_pos_to_backend(pos.first, pos.second);
    }
    return 0;
}
