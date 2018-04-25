import abc
import numpy
import subprocess
import util
import random
import logging
import renju
import nnet


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


class SLAgent:
    def __init__(self, color, policynet, modelfile=None, name='SL agent'):
        self._policynet = policynet
        self._policynet.init_variables(modelfile)

        self._name = name

        if color == "black":
            self._player = renju.Player.BLACK
        elif color == "white":
            self._player = renju.Player.WHITE
        

    def name(self):
        return self._name

    def policy(self, game):
        board = numpy.zeros((15, 15, 9))
        
        if self._player == renju.Player.BLACK:
            board[..., 8] = 1
            
        player_pos = game.positions(player=self._player)
        opponent_pos = game.positions(player=self._player.another())

        k = 2
        for pos in player_pos[-2:-5:-1]:
            i, j = pos
            board[i, j, k] = 1
            k += 2
            
        k = 3
        for pos in opponent_pos[-2:-5:-1]:
            i, j = pos
            board[i, j, k] = 1
            k += 2

        board[..., 0] = game.board(player=self._player)
        board[..., 1] = game.board(player=self._player.another())

        
        prob = self._policynet.predict(numpy.array([board]))
        prob = prob.reshape((15, 15))

        return prob



if __name__ == "__main__":
    g = renju.Game()
    dummy1 = DummyAgent('black')
    sl = SLAgent('white', nnet.PolicyNet(), modelfile='first_model_3epochs')

    renju.run(dummy1, sl)

