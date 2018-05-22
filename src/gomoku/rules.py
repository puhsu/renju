import enum
import numpy
import gomoku.util


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
        return ' '.join(map(gomoku.util.to_move, self._positions))

    @staticmethod
    def loads(dump):
        game = Game()
        for pos in map(gomoku.util.to_pos, dump.split()):
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

        if not self._result and gomoku.util.check(self._board, pos):
            self._result = self._player
            return

        self._player = self._player.another()

    def undo(self):
        '''
        Undo last move
        '''
        i, j = self.positions().pop()
        self.board()[i, j] = 0
