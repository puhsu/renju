import concurrent.futures
import matplotlib.pylab as plt
import enum
import itertools
import logging
import numpy
import sys
import time
import traceback
import util

import random

import renju
import agent


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

    def player(self):
        return self._player

    def result(self):
        return self._result

    def board(self):
        return self._board

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

def number_shift(n):
    if n >= 100:
        return (0.32, 0.15)
    if n >= 10:
        return (0.22, 0.15)
    return (0.10, 0.15)


class PyPlotUI:
    def __init__(self, black='black', white='white'):
        plt.ion()
        self._board = plt.figure(figsize=(8, 8))

        self._ax = self._board.add_subplot(111)
        self._ax.set_navigate(False)

        self._ax.set_title(f'{black} vs {white}')

        self._ax.set_xlim(-1, Game.width)
        self._ax.set_ylim(-1, Game.height)

        self._ax.set_xticks(numpy.arange(0, Game.width))
        self._ax.set_xticklabels(util.POS_TO_LETTER)

        self._ax.set_yticks(numpy.arange(0, Game.height))
        self._ax.set_yticklabels(numpy.arange(1, Game.height + 1))

        self._ax.grid(zorder=2)

        self._black= self._ax.scatter(
            (),(),
            color = 'black',
            s = 500,
            edgecolors = 'black',
            zorder = 3
        )
        self._white = self._ax.scatter(
            (),(),
            color = 'white',
            s = 500,
            edgecolors = 'black',
            zorder = 3
        )

        self._probs = self._ax.imshow(
            numpy.zeros(Game.shape),
            cmap = 'Reds',
            interpolation = 'none',
            vmin = 0.0,
            vmax = 1.0,
            zorder = 1
        )

        self._board.show()


    def update(self, game, probs):
        board = game.board()

        black_positions = util.list_positions(board, Player.BLACK)
        self._black.set_offsets(black_positions[:, (1, 0)])

        white_positions = util.list_positions(board, Player.WHITE)
        self._white.set_offsets(white_positions[:, (1, 0)])

        self._ax.texts = []
        for n, (i, j) in enumerate(game.positions(), 1):
            shift = number_shift(n)
            self._ax.text(
                j - shift[0],
                i - shift[1],
                str(n),
                color = 'white' if n % 2 else 'black',
                fontsize = 10,
                zorder = 4
            )

        self._probs.set_data(probs / 2 * max(probs.max(), 1e-6))

        self._board.canvas.draw()

        return self

def loop(game, black, white, timeout=None):
    yield game, numpy.zeros(game.shape)

    for agent in itertools.cycle([black, white]):
        if not game:
            break
        
        probs = agent.policy(game)

        pos = numpy.unravel_index(probs.argmax(), game.shape)
        game.move(pos)

        yield game, probs


def run_test(black, white, timeout=None):
    game = Game()
    ui = PyPlotUI(black.name(), white.name())

    try:
        for game, probs in loop(game, black, white, timeout):
            ui.update(game, probs)

    except:
        _, e, tb = sys.exc_info()
        print(e)
        traceback.print_tb(tb)
        return game.player().another()

    return game.result()



def run(black, white, max_move_n=60, timeout=10):
    game = Game()

    try:
        for game, _ in loop(game, black, white, timeout):
            logging.debug(game.dumps() + '\n' + str(game.board()))
            if game.move_n() >= max_move_n:
                break

    except:
        logging.error('Error!', exc_info=True, stack_info=True)
        return game.player().another(), game.dumps()

    return game.result(), game.dumps()

if __name__ == "__main__":
    g = renju.Game()
    Dummy = agent.DummyAgent('black')

    ui = PyPlotUI()
    ui.update(g, Dummy.policy(g))
