import abc
import random
import keras
import numpy
import copy
import subprocess

import gomoku.util
import gomoku.rules
import time


class Agent(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_pos(self, game):
        '''Return position on board'''


    @abc.abstractmethod
    def name(self):
        '''return name of agent'''


    @abc.abstractmethod
    def is_human(self):
        '''human predicat'''




class DummyAgent(Agent):
    def __init__(self, color=gomoku.rules.Player.BLACK, name='Random'):
        self._name = name
        self._color = color


    def name(self):
        return self._name


    def color(self):
        return self._color


    def is_human(self):
        return False


    def get_pos(self, game):
        positions = gomoku.util.list_positions(game._board, gomoku.rules.Player.NONE)
        pos = tuple(random.choice(positions))
        return pos




class HumanAgent(Agent):
    def __init__(self, color=gomoku.rules.Player.BLACK, name='Human'):
        self._name = name
        self._color = color
        self.pos = None


    def name(self):
        return self._name


    def color(self):
        return self._color


    def is_human(self):
        return True


    def get_pos(self, game):
        return self.pos



class BackendAgent(Agent):
    def __init__(self, backend, name='BackendAgent', color=gomoku.rules.Player.BLACK, **kvargs):
        self._name = name
        self._color = color
        self._backend = subprocess.Popen(
            backend.split(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            **kvargs
        )

    def name(self):
        return self._name

    def color(self):
        return self._color

    def is_human(self):
        return False

    def terminate(self):
        self._backend.terminate()

    def send_game_to_backend(self, game):
        data = game.dumps().encode()
        print("sending string:", data)
        self._backend.stdin.write(data + b'\n')
        self._backend.stdin.flush()

    def wait_for_backend_move(self):
        data = self._backend.stdout.readline().rstrip()
        print(data)
        return data.decode()

    def get_pos(self, game):
        self.send_game_to_backend(game)
        pos = gomoku.util.to_pos(self.wait_for_backend_move())
        return pos


class SLAgent(Agent):
    def __init__(self, modelfile, color=gomoku.rules.Player.BLACK, name='SL agent'):
        self._name = name
        self._color = color
        self._model = keras.models.load_model(modelfile)


    def name(self):
        return self._name


    def color(self):
        return self._color


    def is_human(self):
        return False


    def get_pos(self, game):
        state = numpy.zeros((15, 15, 4))
        if self._color == gomoku.rules.Player.BLACK:
            state[..., 2] = 1

        player_positions = game.positions(player=self._color)
        opponent_positions = game.positions(player=self._color.another())

        state[..., 0] = game.board(player=self._color)
        state[..., 1] = game.board(player=self._color.another())

        input_board = numpy.array([state])
        prob = self._model.predict(input_board)[0]



        while True:
            p = numpy.random.choice(numpy.arange(15 * 15), p=prob)
            pos = (p // 15, p % 15)
            if game.is_possible_move(pos):
                return pos
            

class Node:
    def __init__(self, pos, value):
        self.pos = pos      # action that we took to get to this node
        self.value = value  # max sum log(probability) from this node
        self.max_child_value = -numpy.inf

        # children info
        self.children = None
        self.probabilities = None

    def is_leaf(self):
        return self.children is None
 

class TreeAgent(Agent):
    """
    Agent that plays few steps ahead with policy network 
    and chooses most probable path
    """
    def __init__(self, modelfile, max_depth=6, max_actions=4, num_iters=100, color=gomoku.rules.Player.BLACK, name='Simple tree'):
        self._name = name
        self._color = color

        self.model = keras.models.load_model(modelfile)
        self.max_depth = max_depth
        self.max_actions = max_actions
        self.num_iters = num_iters
        self.game = None
        self.count_nnet = 0


    def name(self):
        return self._name


    def color(self):
        return self._color


    def is_human(self):
        return False


    def get_pos(self, game):
        """
        return most probable actions from current state
        """
        self.count_nnet = 0;
        self.count_nodes = 1;
        beg = time.time()
        root = Node(None, value=1)
        self.game = copy.deepcopy(game)

        for i in range(self.num_iters):
            self.search(cur=root, depth=0)

        # TODO
        actions = sorted(root.children, key=lambda item: item.value, reverse=True)
        print('Count of nnet runs =', self.count_nnet)
        print('Time elapsed', time.time() - beg, 'seconds')
        return actions[0].pos


    def search(self, cur, depth):
        """
        return action to take from this node
        """
 
        if depth == self.max_depth:
            return

        # expand tree
        if cur.is_leaf():
            state = self.game.state()
            probs = self.model.predict(state.reshape((1, 15, 15, 4))).reshape((15, 15))
            self.count_nnet += 1;

            # get only valid nodes
            valid_actions = probs * self.game.valid()
            
            # create max_actions children
            idx = valid_actions.ravel().argsort()[::-1][:self.max_actions]
            actions = numpy.column_stack(numpy.unravel_index(idx, (15, 15)))
            self.count_nodes += len(actions)

            # normalize probabilities
            probabilities = numpy.array([valid_actions[i, j] for (i, j) in actions])
            prob_sum = numpy.sum(probabilities)
            if prob_sum:
                probabilities /= prob_sum
            else:
                print('All probabilities are zeros. Do something!')
                exit(1)

            cur.children = [Node((i, j), v) for (i, j), v in zip(actions, numpy.log(probabilities))]
            cur.probabilities = probabilities



        # make a move
        child = numpy.random.choice(cur.children, p=cur.probabilities)
        self.game.move(child.pos)
        self.search(child, depth + 1)
        self.game.undo()

        if depth and child.value > cur.max_child_value:
            cur.value += child.value - cur.max_child_value
            cur.max_child_value = child.value




class MCTSAgent:
    def __init__(self):
        pass

    def name(self):
        return self._name

    def color(self):
        return self._color

    def is_human(self):
        return False

    def get_pos(self, game):
        pass

    def search(self):
        pass
