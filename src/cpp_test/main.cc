#include <iostream>

#include "util.h"
#include "rules.h"


int main(int argc, char *argv[]) {
    Session *session;
    load_model("../models/model03tf.pb", &session);


    // test Game object 
    std::cout << "\nTest of Game object\n" << std::endl;
    Game game;
    game.move(7, 7);
    game.move(7, 8);
    game.move(7, 9);
    game.move(0, 0);
    game.move(6, 7);
    game.move(6, 8);
    game.move(5, 7);
    game.move(14, 14);
    game.move(4, 7);
    game.move(9, 9);
    game.move(8, 8);
    game.move(0, 1);
    game.get_state();


    std::cout << "Result: " << game.get_result() << std::endl;
    std::cout << "Player: " << game.get_player() << std::endl;
    std::cout << "Move N: " << game.move_n() << std::endl;
    std::cout << "Last Pos: " << game.last_pos().first << " " << game.last_pos().second << std::endl;
    std::cout << "Bool: " << !game << std::endl;
    game.print_board();


    std::cout << run_model(session, game.get_state()) << std::endl;
    return 0;
}
