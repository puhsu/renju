#include <iostream>
#include <limits>

#include "rules.h"


struct Node {
    Node(pos_t _pos, float _value) 
        : pos(_pos)
        , value(_value)
    {
        std::vector<Node> children;
        std::vector<float> probabilities;
        float max_child_values = std::numeric_limits<float>::min();
    }

    bool is_leaf() const {
        return !children.size();
    }
}


class Tree {
private:
    void search(Node cur, int depth) {
        
    }

public:
    Tree(char *modelfile, 
         int _max_depth, 
         int _max_actions, 
         int _num_iters, 
         Color _color) {
        : max_depth(_max_depth)
        , max_actions(_max_actions)
        , num_iters(_num_iters)
        
        Node root = new Node({-1, -1}, 1);
        
    }

    ~Tree() {

    }

    pos_t get_pos(Game game) {
        pos_t pos = {7, 7};
        return pos;
    }
};


int main(int argc, char *argv[]) {
    Session *session;
    load_model("../models/model03tf.pb", &session);

    Game game;
    game.move(7, 7);
    game.move(7, 8);
    game.move(9, 10);
    game.move(12, 12);
    game.move(14, 14);
    game.move(5, 8);

    for (int i = 0; i < 300; ++i) {
        auto prob = run_model(session, game.get_state());
        // std::cout << std::endl << prob << std::endl;
    }

    // Free any resources used by the session
    Status status = session->Close();
    if (!status.ok()) {
        std::cout << status.ToString() << "\n";
        exit(1);
    }

    return 0;
}
