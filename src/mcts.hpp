//
// Implementation of Monte Carlo Tree Search algorithm
//

#ifndef GOMOKU_MCTS_H
#define GOMOKU_MCTS_H


/*
 * Class that represents one node in the MCTS
 * it holds all the information about current
 * tree node and provides functions to expand
 * and select nodes in MCTS.
 * */

#include <vector>
#include <memory>
#include <cassert>
#include <random>
#include <chrono>
#include <limits>


#include "rules.hpp"
#include "util.hpp"

int created_count = 0;
int deleted_count = 0;

class TreeNode : public std::enable_shared_from_this<TreeNode> {
private:
    friend class MCTS;                                 // give acces to nodes private fields
    std::vector<std::shared_ptr<TreeNode>> children;   // Vector of children (nullptr by default) only valid ones
    std::vector<float> probabilities;                  // Normalized probabilities of actions
    std::weak_ptr<TreeNode> parent;                    // Pointer to the parent node
    float value;                                       // State value function
    bool leaf;                                         // true if node doesn't have children yet
    int visits_count;                                  // number of node visits in rollout stage (starting node counts)
    Color color;                                       // Color of player who's turn it is
public:
    /*
     *  Constructor for creating a Node
     */
    TreeNode(Color c, std::shared_ptr<TreeNode> p = nullptr)
            : value(0)
            , leaf(true)
            , visits_count(0)
            , color(c)
            , parent(p)
    {
        children.resize(225);
        probabilities.resize(225);
    }

    ~TreeNode() {
        ++deleted_count;
    }

    bool is_leaf() const {
        return leaf;
    }

    std::shared_ptr<TreeNode> get_parent() {
        return parent.lock();
    }

    float get_value() const {
        return value / (1 + visits_count);
    }

    float get_reward() const {
        return value;
    }

    int get_visits_count() const {
        return visits_count;
    }


    /*
     * Update value and visits count.
     * Used in backpropagation of reward
     * after evaluation in MCTS
     * */
    void update(float reward) {
        value += reward;
        ++visits_count;
    }


    /*
     * Generate new childs based on the given game state.
     * State is passed from within MCTS class while
     * performing simulation
     * */
    void expand(const Game& game, tensorflow::Session *policy) {
        if (!is_leaf()) {
            return;
        }

        leaf = false;
        auto pred = run_model(policy, game.get_state());
        Eigen::array<Eigen::DenseIndex, 1> dims{{15 * 15}};
        Eigen::Tensor<float, 1, Eigen::RowMajor> flat_pred = pred.reshape(dims);

        float probabilities_sum = 0.0;
        for (int i = 0; i < 15 * 15; ++i) {
            if (game.is_possible_move(i / 15, i % 15)) {
                probabilities[i] = flat_pred(i);
                probabilities_sum += flat_pred(i);
            }
        }

        if (probabilities_sum) {
            for (int i = 0; i < 15 * 15; ++i) {
                probabilities[i] /= probabilities_sum;
            }
        } else {
            std::cerr << "[Bernard] Sum of probabilities is 0. FUCKUP!!!" << std::endl;
        }
    }


    /*
     * Select action from current node according
     * to the upper confidence bound: argmax{Val + Prob * Ns / (1 + Nsa)}
     * Precondition: not called from leaf nodes
     * */
    int select_ucb(const Game& game) {
        assert(!is_leaf());

        int selected;
        float best_value = -std::numeric_limits<float>::max();

        for (int i = 0; i < 225; ++i) {
            // check that we have such child
            if (game.is_possible_move(i / 15, i % 15)) {
                // if no such child get default for value and count
                // othervise retrieve this values from child
                int child_value;
                int child_count;
                if (children[i] == nullptr) {
                    child_value = 0;
                    child_count = 0;
                } else {
                    child_value = children[i]->get_value();
                    child_count = children[i]->get_visits_count();
                }

                // calculate upper confidence bound
                float ucb = child_value + probabilities[i] * (1e-6 + std::sqrt(get_visits_count())) / (1 + child_count);

                //std::cout << "UCB=" << ucb << "\n";
                if (ucb > best_value) {
                    selected = i;
                    best_value = ucb;
                }
            }
        }

        // if this is the first time
        // create a child (do this here to save memory)
        if (children[selected] == nullptr) {
            children[selected] = std::make_shared<TreeNode>(another(color), shared_from_this());
            ++created_count;
        }

        return selected;
    }
};



class MCTS {
private:
    std::shared_ptr<TreeNode> root;    // curreent root node will change it overtime, but the subtree will remain unchanged through the game
    std::chrono::milliseconds timeout; // time limit to make one move
    int rollout_depth;
    Game state;                        // copy of the game state provided by the backend
    Session *policy_session;
    Session *rollout_session;

    float get_reward(Color node_color, Color winner) const {
        if (winner == NONE) {
            return 0;
        } else if (node_color == winner) {
            return -1;
        } else {
            return 1;
        }
    }

    /*
     * Run MCTS simulations for timeout milliseconds
     * */
    void search() {
        std::chrono::system_clock::time_point old=std::chrono::system_clock::now();
        // do simulations while have time
        while (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now()-old) < timeout) {
            Game simulation_state = state;

            // Selection

            std::shared_ptr<TreeNode> selected = root;
            while (!selected->is_leaf()) {
                int action = selected->select_ucb(simulation_state);
                int i = action / 15, j = action % 15;
                if (action == -1) {
                    std::cerr << "[Bernard] Couldn't select action. FUCKUP!!!" << std::endl;
                }
                simulation_state.move(i, j);
                selected = selected->children[action];
            }

            // check termination
            if (!simulation_state) {
                backprop(selected, simulation_state.get_result());
                continue;
            }


            // Expansion

            selected->expand(simulation_state, policy_session);
            int child = selected->select_ucb(simulation_state);
            selected = selected->children[child];
            simulation_state.move(child / 15, child % 15);


            // Evaluation

            Color winner = rollout(simulation_state);
            backprop(selected, winner);
        }
    }


    /*
     * Update all visits count and values up to the root
     * */
    void backprop(std::shared_ptr<TreeNode> node, Color winner) {
        while (node->get_parent() != nullptr) {
            node->update(get_reward(node->color, winner));
            node = node->get_parent();
        }
        node->update(get_reward(node->color, winner));
    }


    Color rollout(Game state) {
        int depth = 0;
        while (state && depth < rollout_depth) {
            ++depth;
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

            // select action randomly with probabilities[]
            std::random_device rd;
            std::mt19937 generator(rd());
            std::discrete_distribution<int> distribution(prob.begin(), prob.end());
            int ind = distribution(generator);
            state.move(actions[ind] / 15, actions[ind] % 15);
        }

        return state.get_result();
    }

public:
    MCTS(std::chrono::milliseconds t, int d,
         tensorflow::Session *policy,
         tensorflow::Session *rollout)
            : timeout(t)
            , rollout_depth(d)
            , policy_session(policy)
            , rollout_session(rollout)
    {}

    ~MCTS() {
        // Free all resourses used by sessions
        Status status = policy_session->Close();
        if (!status.ok()) {
            std::cerr << status.ToString() << "\n";
        }

        status = rollout_session->Close();
        if (!status.ok()) {
            std::cerr << status.ToString() << "\n";
        }
    }

    /*
     * Use this function to update tree after opponent has moved
     * given by the new game state
     * */
    void update_state(Game& new_state) {
        state = new_state;
        root = std::make_shared<TreeNode>(state.get_player());
        ++created_count;
    }

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
            }
        }
        return {best_action / 15, best_action % 15};
    }
};


#endif //GOMOKU_MCTS_H

