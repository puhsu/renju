import abc
import random

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
