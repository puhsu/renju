import os
import random
from colorama import Fore
from colorama import Style


board_format = """
=============
| {} | {} | {} |
|---+---+---|
| {} | {} | {} |
|---+---+---|
| {} | {} | {} |
=============
"""

class Game():
    def __init__(self, playerX, playerO):
        self.state = ' ' * 9
        self.winner = None
        self.draw = False

        self.playerX = playerX
        self.playerO = playerO
        self.cur_player = 'X'

        # variables to store reward and 
        # state from prev state for O and X
        self.rewardX = 0
        self.rewardO = 0
        self.stateX = ' ' * 9
        self.stateO = ' ' * 9
        self.actionX = ' ' * 9
        self.actionO = ' ' * 9

        # variables to store some game statistics
        self.winX_count = 0
        self.winO_count = 0
        self.draw_count = 0
        self.games_count = 0

    def draw_board(self):
        """
        Prints board with numbers of empty cells for easier human interaction
        """
        num = 1
        print_board = []
        for c in self.state:
            if c == ' ':
                print_board.append(num)
                num += 1
            elif c == 'X':
                print_board.append(f'{Fore.GREEN}' + c + f'{Style.RESET_ALL}')
            elif c == 'O':
                print_board.append(f'{Fore.RED}' + c + f'{Style.RESET_ALL}')
        print(board_format.format(*print_board))


    def check_winner(self, state):
        """
        Checks if current state is winnig state for some player and
        returns 'X', 'O' or None if crosses, zeros or nobody wins 
        respectevely
        """
        combinations = [(0,1,2), (3,4,5),
                        (6,7,8), (0,3,6),
                        (1,4,7), (2,5,8),
                        (0,4,8), (2,4,6)]
        for comb in combinations:
            line_state = state[comb[0]] + state[comb[1]] + state[comb[2]]
            if line_state == 'XXX':
                self.winner = 'X'
            if line_state == 'OOO':
                self.winner = 'O'
                
        return self.winner is not None

                     
    def check_draw(self, state):
        filled = ' ' not in state
        no_winner = not self.check_winner(state)
        self.draw = filled and no_winner
        return filled and no_winner

    def check_terminal(self, state):
        return ' ' not in state


    def make_turn(self, train=False, step=None):
        """
        Tries to make move asking agent for it's action and for SARSA, Qlearning
        gives reward if in trainig mode.
        
        returns: 
          0 -- if game is still going
          1 -- if game stopped after this turn
        """
        if self.cur_player == 'X':
            if not self.check_terminal(self.state):
                new_state = self.playerX.play(self.state, train=train)
                self.check_winner(new_state)
                self.check_draw(new_state)
            else:
                new_state = self.state
            reward = 0


            if self.winner == 'X':
                reward = 1
            if self.winner == 'O':
                reward = -1


            if train and step != 0:
                self.playerX.update(old_state=self.stateX,
                                    old_action=self.actionX,
                                    new_state=self.state,
                                    reward=reward)
            
            if train:
                self.stateX = self.state
                self.actionX = new_state
                self.rewardX = reward
            self.cur_player = 'O'
            
        elif self.cur_player == 'O':
            if not self.check_terminal(self.state):
                new_state = self.playerO.play(self.state, train=train)
                self.check_winner(new_state)
                self.check_draw(new_state)
            else:
                new_state = self.state
            reward = 0
            

            if self.winner == 'O':
                reward = 1
            elif self.winner == 'X':
                reward = -1

            if train and step != 1:
                self.playerO.update(old_state=self.stateO,
                                    old_action=self.actionO,
                                    new_state=self.state, 
                                    reward=reward)

            # update allstat's ang 
            # get ready for next step
            if train:
                self.stateO = self.state
                self.actionO = new_state
                self.rewardO = reward
            self.cur_player = 'X'

        self.state = new_state
        if not train and (self.winner or self.draw):
            return 1
        if self.check_terminal(self.stateO) or self.check_terminal(self.stateX):
            # get stats
            if self.winner == 'X':
                self.winX_count += 1
            elif self.winner == 'O':
                self.winO_count += 1
            elif self.draw:
                self.draw_count += 1
            self.games_count += 1
            return 1
        else:
            return 0

    def train_agents(self, iterations=1000, verbose=False):
        """
        Trains players by competing them against each other
        returns:
          agentX -- agent which plays crosses
          agentO -- agent which plays zeros
        """
        for i in range(iterations):
            finished = False
            j = 0
            while not finished:
                finished = self.make_turn(train=True, step=j)
                j += 1
            
            self.state = ' ' * 9
            self.winner = None
            self.draw = False
            self.cur_player = 'X'


            self.rewardX = 0
            self.rewardO = 0
            self.stateX = ' ' * 9
            self.stateO = ' ' * 9
            self.actionX = ' ' * 9
            self.actionO = ' ' * 9
        
            
            if verbose and (i + 1) % 10000 == 0:
                x_ratio = self.winX_count / self.games_count
                o_ratio = self.winO_count / self.games_count
                draw_ratio = self.draw_count / self.games_count
                print('Iteration {}. X = {:.3f} | O = {:.3f} | Draw = {:.3f}'.format(i+1, x_ratio, o_ratio, draw_ratio))

                self.winX_count = 0
                self.winO_count = 0
                self.draw_count = 0
                self.games_count = 0
                # print(self.playerX.Q.values())
        return self.playerX, self.playerO

    def interactive_game(self):
        playing = True
        while playing:
            finished = False
            while not finished:
                # print
                humanO = self.cur_player == 'O' and self.playerO.ishuman
                humanX = self.cur_player == 'X' and self.playerX.ishuman

                if humanO or humanX:
                    os.system('clear')
                    if self.cur_player == 'X':
                        print(f'Turn for {Fore.GREEN}' + '{}'.format(self.cur_player) + f'{Style.RESET_ALL}')
                    else:
                        print(f'Turn for {Fore.RED}' + '{}'.format(self.cur_player) + f'{Style.RESET_ALL}')
                    self.draw_board()

                finished = self.make_turn(train=False)

            os.system('clear')
            # Inform about results
            if self.draw:
                print("It's a draw)")
            if self.winner == 'O':
                print("O player wins")
            if self.winner == 'X':
                print("X player wins")

            self.draw_board()
            playing = input('Play again [y]? ') == 'y'

            self.state = ' ' * 9
            self.winner = None
            self.draw = False
            self.cur_player = 'X'




class Human():
    def __init__(self, side='X'):
        self.ishuman = True
        self.side = side

    def play(self, state, train=False):
        cell_map = dict()
        num = 1
        num_list = []
        for i, c in enumerate(state):
            if c == ' ':
                cell_map[num] = i
                num_list.append(num)
                num += 1

        choice = cell_map[int(input('Choose cell from {}: '.format(num_list)))]
        state = state[:choice] + self.side + state[choice+1:]
        return state



class Qlearner():
    def __init__(self, alpha=.5, eps=.01, discount=.6, side='X'):
        self.Q = dict()
        self.alpha = alpha
        self.eps = eps
        self.discount = discount
        self.side = side
        self.ishuman = False

    def check_terminal(self, state):
        return ' ' not in state

    def allowed_moves(self, state):
        """
        Finds all allowed moves from current states and 
        returns:
          actions -- list of allowed actions in current state
        """
        actions = []
        for i in range(len(state)):
            if state[i] == ' ':
                actions.append(state[:i] + self.side + state[i+1:])
        return actions


    def play(self, state, train=False):
        """
        Choose current move from state using current Q function
        """
        actions = self.allowed_moves(state)
        actionsQ = dict() 
        for action in actions:
            if (state, action) not in self.Q:
                self.Q[state, action] = 0
            actionsQ[action] = self.Q[state, action]
        
        
        if train and random.random() < self.eps:
            return random.choice([a for a, v in actionsQ.items()])
        else:
            maxQ = max(actionsQ.values())
            return random.choice([a for a, v in actionsQ.items() if v == maxQ])


    def update(self, old_state, old_action, new_state, reward):
        """
        Update Q function
        """
        actions = self.allowed_moves(new_state)
        actions_values = []
        for action in actions:
            if (new_state, action) not in self.Q:
                self.Q[new_state, action] = 0
            actions_values.append(self.Q[new_state, action])
        if self.check_terminal(new_state):
            maxQ = 0
        else:
            maxQ = max(actions_values)

        self.Q[old_state, old_action] += self.alpha * (reward + self.discount * maxQ - \
                                                       self.Q[old_state, old_action])
        
if __name__ == '__main__':
    agentX = Qlearner(alpha=1.0, eps=.02, discount=1.0, side='X')
    agentO = Qlearner(alpha=1.0, eps=.01, discount=1.0, side='O')
    game = Game(agentX, agentO)
    game.state

    humanX = Human(side='X')
    import os
import random
from colorama import Fore
from colorama import Style


board_format = """
=============
| {} | {} | {} |
|---+---+---|
| {} | {} | {} |
|---+---+---|
| {} | {} | {} |
=============
"""

class Game():
    def __init__(self, playerX, playerO):
        self.state = ' ' * 9
        self.winner = None
        self.draw = False

        self.playerX = playerX
        self.playerO = playerO
        self.cur_player = 'X'

        # variables to store reward and 
        # state from prev state for O and X
        self.rewardX = 0
        self.rewardO = 0
        self.stateX = ' ' * 9
        self.stateO = ' ' * 9
        self.actionX = ' ' * 9
        self.actionO = ' ' * 9

        # variables to store some game statistics
        self.winX_count = 0
        self.winO_count = 0
        self.draw_count = 0
        self.games_count = 0

    def draw_board(self):
        """
        Prints board with numbers of empty cells for easier human interaction
        """
        num = 1
        print_board = []
        for c in self.state:
            if c == ' ':
                print_board.append(num)
                num += 1
            elif c == 'X':
                print_board.append(f'{Fore.GREEN}' + c + f'{Style.RESET_ALL}')
            elif c == 'O':
                print_board.append(f'{Fore.RED}' + c + f'{Style.RESET_ALL}')
        print(board_format.format(*print_board))


    def check_winner(self, state):
        """
        Checks if current state is winnig state for some player and
        returns 'X', 'O' or None if crosses, zeros or nobody wins 
        respectevely
        """
        combinations = [(0,1,2), (3,4,5),
                        (6,7,8), (0,3,6),
                        (1,4,7), (2,5,8),
                        (0,4,8), (2,4,6)]
        for comb in combinations:
            line_state = state[comb[0]] + state[comb[1]] + state[comb[2]]
            if line_state == 'XXX':
                self.winner = 'X'
            if line_state == 'OOO':
                self.winner = 'O'
                
        return self.winner is not None

                     
    def check_draw(self, state):
        filled = ' ' not in state
        no_winner = not self.check_winner(state)
        self.draw = filled and no_winner
        return filled and no_winner

    def check_terminal(self, state):
        return ' ' not in state


    def make_turn(self, train=False, step=None):
        """
        Tries to make move asking agent for it's action and for SARSA, Qlearning
        gives reward if in trainig mode.
        
        returns: 
          0 -- if game is still going
          1 -- if game stopped after this turn
        """
        if self.cur_player == 'X':
            if not self.check_terminal(self.state):
                new_state = self.playerX.play(self.state, train=train)
                self.check_winner(new_state)
                self.check_draw(new_state)
            else:
                new_state = self.state
            reward = 0


            if self.winner == 'X':
                reward = 10
            if self.winner == 'O':
                reward = -10
            elif self.draw:
                reward = 1


            if train and step != 0:
                self.playerX.update(old_state=self.stateX,
                                    old_action=self.actionX,
                                    new_state=self.state,
                                    reward=reward)
            
            self.stateX = self.state
            self.actionX = new_state
            self.rewardX = reward
            self.cur_player = 'O'
            
        elif self.cur_player == 'O':
            if not self.check_terminal(self.state):
                new_state = self.playerO.play(self.state, train=train)
                self.check_winner(new_state)
                self.check_draw(new_state)
            else:
                new_state = self.state
            reward = 0
            

            if self.winner == 'O':
                reward = 10
            elif self.winner == 'X':
                reward = -10
            elif self.draw:
                reward = 1

            if train and step != 1:
                self.playerO.update(old_state=self.stateO,
                                    old_action=self.actionO,
                                    new_state=self.state, 
                                    reward=reward)

            # update allstat's ang 
            # get ready for next step
            self.stateO = self.state
            self.actionO = new_state
            self.rewardO = reward
            self.cur_player = 'X'

        self.state = new_state
        if not train and (self.winner or self.draw):
            return 1
        if self.check_terminal(self.stateO) or self.check_terminal(self.stateX):
            # get stats
            if self.winner == 'X':
                self.winX_count += 1
            elif self.winner == 'O':
                self.winO_count += 1
            elif self.draw:
                self.draw_count += 1
            self.games_count += 1
            return 1
        else:
            return 0

    def train_agents(self, iterations=1000, verbose=False):
        """
        Trains players by competing them against each other
        returns:
          agentX -- agent which plays crosses
          agentO -- agent which plays zeros
        """
        for i in range(iterations):
            finished = False
            j = 0
            while not finished:
                finished = self.make_turn(train=True, step=j)
                j += 1
            
            self.state = ' ' * 9
            self.winner = None
            self.draw = False
            self.cur_player = 'X'


            self.rewardX = 0
            self.rewardO = 0
            self.stateX = ' ' * 9
            self.stateO = ' ' * 9
            self.actionX = ' ' * 9
            self.actionO = ' ' * 9
        
            
            if verbose and (i + 1) % 10000 == 0:
                x_ratio = self.winX_count / self.games_count
                o_ratio = self.winO_count / self.games_count
                draw_ratio = self.draw_count / self.games_count
                print('Iteration {}. X = {:.3f} | Y = {:.3f} | Draw = {:.3f}'.format(i+1, x_ratio, o_ratio, draw_ratio))

                self.winX_count = 0
                self.winO_count = 0
                self.draw_count = 0
                self.games_count = 0
                # print(self.playerX.Q.values())
        return self.playerX, self.playerO

    def interactive_game(self):
        playing = True
        while playing:
            finished = False
            while not finished:
                # print
                humanO = self.cur_player == 'O' and self.playerO.ishuman
                humanX = self.cur_player == 'X' and self.playerX.ishuman

                if humanO or humanX:
                    os.system('clear')
                    if self.cur_player == 'X':
                        print(f'Turn for {Fore.GREEN}' + '{}'.format(self.cur_player) + f'{Style.RESET_ALL}')
                    else:
                        print(f'Turn for {Fore.RED}' + '{}'.format(self.cur_player) + f'{Style.RESET_ALL}')
                    self.draw_board()

                finished = self.make_turn(train=False)

            os.system('clear')
            # Inform about results
            if self.draw:
                print("It's a draw)")
            if self.winner == 'O':
                print("O player wins")
            if self.winner == 'X':
                print("X player wins")

            self.draw_board()
            playing = input('Play again [y]? ') == 'y'

            self.state = ' ' * 9
            self.winner = None
            self.draw = False
            self.cur_player = 'X'




class Human():
    def __init__(self, side='X'):
        self.ishuman = True
        self.side = side

    def play(self, state, train=False):
        cell_map = dict()
        num = 1
        num_list = []
        for i, c in enumerate(state):
            if c == ' ':
                cell_map[num] = i
                num_list.append(num)
                num += 1

        choice = cell_map[int(input('Choose cell from {}: '.format(num_list)))]
        state = state[:choice] + self.side + state[choice+1:]
        return state



class Qlearner():
    def __init__(self, alpha=.5, eps=.01, discount=.6, side='X'):
        self.Q = dict()
        self.alpha = alpha
        self.eps = eps
        self.discount = discount
        self.side = side
        self.ishuman = False

    def check_terminal(self, state):
        return ' ' not in state

    def allowed_moves(self, state):
        """
        Finds all allowed moves from current states and 
        returns:
          actions -- list of allowed actions in current state
        """
        actions = []
        for i in range(len(state)):
            if state[i] == ' ':
                actions.append(state[:i] + self.side + state[i+1:])
        return actions


    def play(self, state, train=False):
        """
        Choose current move from state using current Q function
        """
        actions = self.allowed_moves(state)
        actionsQ = dict() 
        for action in actions:
            if (state, action) not in self.Q:
                self.Q[state, action] = 0
            actionsQ[action] = self.Q[state, action]
        
        
        if train and random.random() < self.eps:
            return random.choice([a for a, v in actionsQ.items()])
        else:
            maxQ = max(actionsQ.values())
            return random.choice([a for a, v in actionsQ.items() if v == maxQ])


    def update(self, old_state, old_action, new_state, reward):
        """
        Update Q function
        """
        actions = self.allowed_moves(new_state)
        actions_values = []
        for action in actions:
            if (new_state, action) not in self.Q:
                self.Q[new_state, action] = 0
            actions_values.append(self.Q[new_state, action])
        if self.check_terminal(new_state):
            maxQ = 0
        else:
            maxQ = max(actions_values)

        self.Q[old_state, old_action] += self.alpha * (reward + self.discount * maxQ - \
                                                       self.Q[old_state, old_action])
        
if __name__ == '__main__':    
    humanX = Human(side='X')
    humanO = Human(side='O')


    agentX = Qlearner(alpha=1.0, eps=.01, discount=1.0, side='X')
    agentO = Qlearner(alpha=1.0, eps=.01, discount=1.0, side='O')
    game = Game(agentX, agentO)


    agentX, agentO = game.train_agents(iterations=50000, verbose=True)
    agentX.eps = .5

    game = Game(agentX, agentO)
    agentX, agentO = game.train_agents(iterations=60000, verbose=True)

    game = Game(humanX, agentO)
    game.interactive_game()

    game = Game(agentX, humanO)
    game.interactive_game()
