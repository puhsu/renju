import abc
import numpy
import subprocess
import util
import random
import logging
import renju


class Agent(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def policy(self, game):
        '''Return probabilty matrix of possible actions'''

    @abc.abstractmethod
    def name(self):
        '''return name of agent'''

class HumanAgent(Agent):
    def __init__(self, color, name='Human'):
        self._name = name
        if color == "black":
            self._player = renju.Player.BLACK
        elif color == "white":
            self._player = renju.Player.WHITE

    def name(self):
        return self._name

    def policy(self, game):
        move = input()
        pos = util.to_pos(move)

        probs = numpy.zeros(game.shape)
        probs[pos] = 1.0

        return probs

class DummyAgent(Agent):
    def __init__(self, color, name='Random'):
        self._name = name
        if color == "black":
            self._player = renju.Player.BLACK
        elif color == "white":
            self._player = renju.Player.WHITE

    def name(self):
        return self._name

    def policy(self, game):
        positions = util.list_positions(game._board, renju.Player.NONE)
        pos = tuple(random.choice(positions))

        probs = numpy.zeros(game.shape)
        probs[pos] = 1.0

        return probs

        



if __name__ == "__main__":
    g = renju.Game()
    dummy1 = DummyAgent('black')
    dummy2 = DummyAgent('white')

    renju.run(dummy1, dummy2)

