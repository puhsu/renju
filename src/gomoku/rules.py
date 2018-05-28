import enum
import itertools
import numpy


#
# Utility functions
#

POS_TO_LETTER = 'abcdefghjklmnop'
LETTER_TO_POS = {letter: pos for pos, letter in enumerate(POS_TO_LETTER)}

def to_move(pos):
    return POS_TO_LETTER[pos[1]] + str(pos[0] + 1)

def to_pos(move):
    return int(move[1:]) - 1, LETTER_TO_POS[move[0]]

def list_positions(board, player):
    return numpy.vstack(numpy.nonzero(board == player)).T

#
# Game rules
#

def sequence_length(board, I, J, value):
    length = 0

    for i, j in zip(I, J):
        if board[i, j] != value:
            break
        length += 1

    return length


def check_horizontal(board, pos):
    player = board[pos]
    if not player:
        return False

    i, j = pos
    length = 1

    length += sequence_length(
        board,
        itertools.repeat(i),
        range(j + 1, min(j + Game.line_length, Game.width)),
        player
    )

    length += sequence_length(
        board,
        itertools.repeat(i),
        range(j - 1, max(j - Game.line_length, -1), -1),
        player
    )

    return length >= Game.line_length

def check_vertical(board, pos):
    player = board[pos]
    if not player:
        return False

    i, j = pos
    length = 1

    length += sequence_length(
        board,
        range(i + 1, min(i + Game.line_length, Game.height)),
        itertools.repeat(j),
        player
    )

    length += sequence_length(
        board,
        range(i - 1, max(i - Game.line_length, -1), -1),
        itertools.repeat(j),
        player
    )

    return length >= Game.line_length

def check_main_diagonal(board, pos):
    player = board[pos]
    if not player:
        return False

    i, j = pos
    length = 1

    length += sequence_length(
        board,
        range(i + 1, min(i + Game.line_length, Game.height)),
        range(j + 1, min(j + Game.line_length, Game.width)),
        player
    )

    length += sequence_length(
        board,
        range(i - 1, max(i - Game.line_length, -1), -1),
        range(j - 1, max(j - Game.line_length, -1), -1),
        player
    )

    return length >= Game.line_length

def check_side_diagonal(board, pos):
    player = board[pos]
    if not player:
        return False

    i, j = pos
    length = 1

    length += sequence_length(
        board,
        range(i - 1, max(i - Game.line_length, -1), -1),
        range(j + 1, min(j + Game.line_length, Game.width)),
        player
    )

    length += sequence_length(
        board,
        range(i + 1, min(i + Game.line_length, Game.height)),
        range(j - 1, max(j - Game.line_length, -1), -1),
        player
    )

    return length >= Game.line_length

def check(board, pos):
    if not board[pos]:
        return False

    return check_vertical(board, pos) \
        or check_horizontal(board, pos) \
        or check_main_diagonal(board, pos) \
        or check_side_diagonal(board, pos)


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

    def valid(self):
        return (self._board == Player.NONE).astype(numpy.int8)

    def state(self):
        '''
        return state for current player as numpy array of shape (15, 15, 4)
        '''
        state = numpy.zeros((15, 15, 4))
        state[..., 0] = self.board(player=self.player())
        state[..., 1] = self.board(player=self.player().another())
        if self.player() == Player.BLACK:
            state[..., 2] = 1
        return state

    def positions(self, player=Player.NONE):
        if not player:
            return self._positions

        begin = 0 if player == Player.BLACK else 1
        return self._positions[begin::2]

    def dumps(self):
        return ' '.join(map(to_move, self._positions))

    @staticmethod
    def loads(dump):
        game = Game()
        for pos in map(to_pos, dump.split()):
            game.move(pos)
        return game


    def is_possible_move(self, pos):
        return 0 <= pos[0] < self.height \
            and 0 <= pos[1] < self.width \
            and not self._board[pos]

    def move(self, pos):
        assert self.is_possible_move(pos), f'impossible pos: {pos}'

        self._positions.append(pos)
        self._board[pos] = self._player

        if not self._result and check(self._board, pos):
            self._result = self._player
            return

        self._player = self._player.another()

    def undo(self):
        '''
        Undo last move
        '''
        i, j = self.positions().pop()
        self.board()[i, j] = 0
