#include <iostream>
#include <limits>
#include <chrono>
#include <queue>
#include <random>
#include <memory>
#include <unistd.h>

#include "rules.h"
#include "logger.h"

const std::string project_dir = "/Users/irubachev/Documents/cs/project/renju/src/cpp_test/";
using namespace std::chrono;

// global tf sessions used inside mcts
Session *policy_session;
Session *rollout_session;


// Class that represents one node in the MCTS
// it holds all the information about current 
// tree node and provides functions to expand
// and select nodes in MCTS.

class TreeNode : public std::enable_shared_from_this<TreeNode> {
private:
    friend class MCTS;                                 // give acces to nodes private fields
    std::vector<std::shared_ptr<TreeNode>> children;   // Vector of children (nullptr by default) only valid ones
    std::vector<float> probabilities;                  // Normalized probabilities of actions
    std::weak_ptr<TreeNode> parent;                    // Pointer to the parent node
    float value;                                       // State value function
    bool leaf;                                         // true if node doesn't have children yet
    int visits_count;                                  // number of node visits in rollout stage (starting node counts)
    Color color;                                       // color of the current node

    // algorithm parameters
    const float EPS = 1e-6;
    const float Cpuct = 1;
    
public:
    /*
     *  Default constructor for creating a Node 
     */
    TreeNode(Color c, std::shared_ptr<TreeNode> p = nullptr) 
        : value(0)
        , leaf(true)
        , visits_count(0)
        , parent(p)
        , color(c)
    {
        children.resize(225);
        probabilities.resize(225);
    }

    // TODO: copy constructor

    std::shared_ptr<TreeNode> get_parent() {
        return parent.lock();
    }


    bool is_leaf() const {
        return leaf;
    }


    // Update value and visits count
    // used in backpropagation of reward
    // after evaluation in MCTS

    void update(float reward) {
        value += reward;
        ++visits_count;
    }


    float get_value() const {
        return value / visits_count;
    }


    int get_visits_count() const {
        return visits_count;
    }

    
    // Generate new childs based on the given game state
    // State is passed from within MCTS class while 
    // performing simulation

    void expand(const Game& game) {
        if (!is_leaf()) {
            return;
        }

        leaf = false;
        // run policy network
        auto pred = run_model(policy_session, game.get_state());
        
        // flatten output
        Eigen::array<Eigen::DenseIndex, 1> dims{{15 * 15}};
        Eigen::Tensor<float, 1, Eigen::RowMajor> flat_pred = pred.reshape(dims);

        // select only valid actions
        // and create child nodes
        float prob_sum = 0.0;
        for (int i = 0; i < 15 * 15; ++i) {
            if (game.is_possible_move(i / 15, i % 15)) {
                children[i] = std::make_shared<TreeNode>(another(color), shared_from_this());
                probabilities[i] = flat_pred(i);
                prob_sum += flat_pred(i);
            }
        }

        // renormalize probabilities
        for (int i = 0; i < 15 * 15; ++i) {
            probabilities[i] /= prob_sum;
        }
    }


    // Select action from current node according 
    // to the upper confidence bound: argmax Val + Prob/N
    // Note: not called from leaf nodes

    int select_ucb() const {
        // ensure it isn't leaf node
        assert(!is_leaf());

        int selected = 0;
        float best_value = std::numeric_limits<float>::min();

        for (int i = 0; i < 225; ++i) {
            // check that we have such child
            if (children[i]) {
                // calculate upper confidence bound
                float u = Cpuct * probabilities[i] * std::sqrt(get_visits_count()) 
                          / (1 + children[i]->get_visits_count());

                float ucb = children[i]->value + u;

                if (ucb > best_value) {
                    selected = i;
                    best_value = ucb;
                }
            }
        }

        return selected;
    }


    // UTILITY/DEBUG FUNCTIONS
    /*
     *  Count nodes in subtree with this node as its root.
     */
    int count_of_nodes() const {
        if (is_leaf()) {
            return 1;
        }

        int res = 1;
        for (int i = 0; i < 225; ++i) {
            if (children[i]) {
                res += children[i]->count_of_nodes();
            }
        }
        return res;
    }
};




class MCTS {
private:
    std::shared_ptr<TreeNode> root;    // curreent root node will change it overtime, but the subtree will remain unchanged through the game
    milliseconds timeout;              // time limit to make one move
    Game state;                        // copy of the game state provided by the backend

    float get_reward(Color node_color, Color winner) const {
        if (winner == NONE) {
            return 0;
        } else if (node_color == winner) {
            return 1;
        } else {
            return -1;
        }
    }

    // search procedure iteration itself
    void search() {
        system_clock::time_point old=system_clock::now();
        int sim_count = 0;
        // do simulations while have time
        while (duration_cast<milliseconds>(system_clock::now()-old) < timeout) {
            ++sim_count;
            // Create deep copy of game state to do our work in it
            Game simulation_state = state;

            /*
             *  Selection
             */
        
            // goto leaf node
            std::shared_ptr<TreeNode> selected = root;
            while (!selected->is_leaf()) {
                int action = selected->select_ucb();
                int i = action / 15, j = action % 15;

                simulation_state.move(i, j);
                selected = selected->children[action];
            }

            // check termination
            if (!simulation_state) {
                backprop(selected, simulation_state.get_result());
                continue;
            }

            
            /*
             *  Expansion
             */
            selected->expand(simulation_state);

            /*
             *  Evaluation
             */
            Color winner = rollout(simulation_state);
            backprop(selected, winner);
        }
        //std::cout << "Finished simulations: " << sim_count << std::endl;
    }

    // update all counts and values up the tree
    void backprop(std::shared_ptr<TreeNode> node, Color winner) {
        while (node->get_parent() != nullptr) {
            node->update(get_reward(node->color, winner));
            node = node->get_parent();
        }
        node->update(get_reward(node->color, winner));
    }


    // do rollout and return the winner
    Color rollout(Game state) {
        while (state) {
            auto pred = run_model(rollout_session, state.get_state());

            // flatten output
            Eigen::array<Eigen::DenseIndex, 1> dims{{15 * 15}};
            Eigen::Tensor<float, 1, Eigen::RowMajor> flat_pred = pred.reshape(dims);
            
            // select only valid actions
            std::vector<int> actions;
            std::vector<float> prob;
            for (int i = 0; i < 15 * 15; ++i) {
                if (state.is_possible_move(i / 15, i % 15)) {
                    actions.push_back(i);
                    prob.push_back(flat_pred(i));
                }
            }

            // get top k actions
            int k = 5;
            std::vector<int> top_actions;
            std::vector<float> top_prob;
            
            std::priority_queue<std::pair<float, int>> q;
            for (int i = 0; i < prob.size(); ++i) {
                q.push({prob[i], i});
            }
        
            for (int i = 0; i < k; ++i) {
                int ki = q.top().second;
                top_actions.push_back(actions[ki]);
                top_prob.push_back(prob[ki]);
                q.pop();
            }

            // select action randomly with probabilities given by top_prob
            std::random_device rd;
            std::mt19937 generator(rd());
            std::discrete_distribution<int> distribution(top_prob.begin(), top_prob.end());
            int ind = distribution(generator);
            state.move(top_actions[ind] / 15, top_actions[ind] % 15);
        }

        return state.get_result();
    }

public:
    int expanded = 0;

    MCTS(milliseconds t) 
        : timeout(t)
    {
    }

    // Use this function to update tree after opponent has moved
    // param last opponent move
    void update_state(Game new_state) {
        state = new_state;
        pos_t opponent_pos;
        int opponent_action;
        // get opponent move from new_state
        if (state.move_n()) {
            opponent_pos = state.last_pos();
            opponent_action = opponent_pos.first * 15 + opponent_pos.second;
            //std::cout << "(" << opponent_pos.first << ", " << opponent_pos.second << ")\n";
        }

        // we have this node in our tree
        // just update the root and save subtree
        if (state.move_n() && root->children[opponent_action]) {
            root = root->children[opponent_action];
        } else {
            root = std::make_shared<TreeNode>(state.get_player());
        }
    }

    // do simulations and return best position
    pos_t get_pos() {
        search();

        int best_action;
        int max_count = std::numeric_limits<int>::min();

        for (int i = 0; i < 225; ++i) {
            if (root->children[i]) {
                int cur_count = root->children[i]->get_visits_count();
                if (cur_count > max_count) {
                    best_action = i;
                    max_count = cur_count;
                }
                //std::cout << cur_count << " (" << i/15 << ", " << i%15 << ")\n";
            }
        }
        return {best_action / 15, best_action % 15};
    }
};


int main(int argc, char *argv[]) {
    // load models
    load_model(project_dir + "models/model.policy.04.pb", &policy_session);
    load_model(project_dir + "models/model.policy.04.pb", &rollout_session);

    ige::FileLogger log ("1.0", "logfile.txt");

    MCTS tree(milliseconds{1000});

    while (true) {
        Game state = backend::wait_for_game_update();
        tree.update_state(state);
        pos_t pos = tree.get_pos();
        backend::send_pos_to_backend(pos.first, pos.second);
        


        //Game game = backend::wait_for_game_update();
        //tree.update_state(game);
        //pos_t pos = tree.get_pos();
        //std::cout << "suka imtashrtoierst\n";
        //backend::move(pos.first, pos.second);
    }

    // Free any resources used by the sessions
    Status status = policy_session->Close();
    if (!status.ok()) {
        std::cerr << status.ToString() << "\n";
        exit(1);
    }

    status = rollout_session->Close();
    if (!status.ok()) {
        std::cerr << status.ToString() << "\n";
        exit(1);
    }

    return 0;
}
