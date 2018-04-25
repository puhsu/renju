import enum
import itertools
import logging
import numpy
import sys
import time
import traceback
import util


class Player(enum.IntEnum):
    NONE = 0
    BLACK = -1
    WHITE = 1

    def another(self):
        return Player(-self)

    def __repr__(self):
        if self == Player.BLACK:
            return 'black'
        elif self == Player.WHITE:
            return 'white'
        else:
            return 'none'

    def __str__(self):
        return self.__repr__()

class Game:
    width, height = 15, 15
    shape = (width, height)
    line_length = 5

    def __init__(self):
        self._result = Player.NONE
        self._player = Player.BLACK
        self._board = numpy.full(self.shape, Player.NONE, dtype=numpy.int8)
        self._positions = list()

    def __bool__(self):
        return self.result() == Player.NONE and \
            len(self._positions) < self.width * self.height

    def move_n(self):
        return len(self._positions)

    def last_pos(self):
        return self._positions[-1]

    def player(self):
        return self._player

    def result(self):
        return self._result

    def board(self, player=Player.NONE):
        if not player:
            return self._board
        return (self._board == player).astype(numpy.int8)

    def positions(self, player=Player.NONE):
        if not player:
            return self._positions

        begin = 0 if player == Player.BLACK else 1
        return self._positions[begin::2]

    def dumps(self):
        return ' '.join(map(util.to_move, self._positions))

    @staticmethod
    def loads(dump):
        game = Game()
        for pos in map(util.to_pos, dump.split()):
            game.move(pos)
        return game


    def is_posible_move(self, pos):
        return 0 <= pos[0] < self.height \
            and 0 <= pos[1] < self.width \
            and not self._board[pos]

    def move(self, pos):
        assert self.is_posible_move(pos), f'impossible pos: {pos}'

        self._positions.append(pos)
        self._board[pos] = self._player

        if not self._result and util.check(self._board, pos):
            self._result = self._player
            return

        self._player = self._player.another()


def loop(game, black, white):
    yield game, numpy.zeros(game.shape)

    for agent in itertools.cycle([black, white]):
        if not game:
            break
        
        probs = agent.policy(game)
        ind = numpy.unravel_index(numpy.argsort(probs, axis=None)[::-1], (15, 15))


        pos = []
        for i in range(15 * 15):
            pos.append((ind[0][i], ind[1][i]))

        for p in pos:
            if game.is_posible_move(p):
                game.move(p)
                break

        yield game, probs


def run_test(black, white, timeout=None):
    game = Game()
    ui = (black.name(), white.name())

    try:
        for game, probs in loop(game, black, white):
            ui.update(game, probs)
            

    except:
        _, e, tb = sys.exc_info()
        print(e)
        traceback.print_tb(tb)
        return game.player().another()

    return game.result(), game



if __name__ == "__main__":
    Dummy1 = agent.DummyAgent('black')
    Dummy2 = agent.DummyAgent('white')
    #run_test(Dummy1, Dummy2)
