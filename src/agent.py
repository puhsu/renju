import abc
import random
import keras
import numpy

import util
import renju


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
    def __init__(self, color=renju.Player.BLACK, name='Random'):
        self._name = name
        self._color = color


    def name(self):
        return self._name


    def color(self):
        return self._color


    def is_human(self):
        return False


    def get_pos(self, game):
        positions = util.list_positions(game._board, renju.Player.NONE)
        pos = tuple(random.choice(positions))
        return pos




class HumanAgent(Agent):
    def __init__(self, color=renju.Player.BLACK, name='Human'):
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




class SLAgent(Agent):
    def __init__(self, modelfile, color=renju.Player.BLACK, name='SL agent'):
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
        if self._color == renju.Player.BLACK:
            state[..., 2] = 1

        player_positions = game.positions(player=self._color)
        opponent_positions = game.positions(player=self._color.another())

        state[..., 0] = game.board(player=self._color)
        state[..., 1] = game.board(player=self._color.another())

        input_board = numpy.array([state])
        prob = self._model.predict(input_board)
        prob = prob.reshape((15, 15))

        ind = numpy.unravel_index(numpy.argsort(prob, axis=None)[::-1], (15, 15))
        pos = []
        for i in range(15 * 15):
            pos.append((ind[0][i], ind[1][i]))

        for p in pos:
            if game.is_possible_move(p):
                return p
