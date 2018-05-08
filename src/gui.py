import tkinter
import tkinter.font
import math

import renju
import agent


class GomokuUI(tkinter.Frame):
    def __init__(self, master=None, black=None, white=None, timeout=0):
        super().__init__(master)
        self.game = renju.Game()
        self._black = black
        self._white = white
        self._player = self._black
        self._timeout = timeout
        # start game
        self.create_widgets()
        self.game_loop()
    

    def create_widgets(self):
        header_text = self._black.name() + ' [black] vs ' + self._white.name() + ' [white]'
        bottom_text = self._black.name() + ' turn now...'

        self.board_canvas = BoardCanvas(height=550, width=530)
        self.header = tkinter.Label(self.master, font=('Monaco', 15), text=header_text)
        self.bottom = tkinter.Label(self.master, font=('Monaco', 15), text=bottom_text, height=2)
        self.board_canvas.bind('<Button-1>', self.read_move)
        self.header.pack()
        self.board_canvas.pack()
        self.bottom.pack()


    def reset_game(self, e):
        if e.char == 'h':
            self.board_canvas.delete('all')
            self.board_canvas.draw_board()
            self.master.unbind('<KeyPress>')
            self.board_canvas.bind('<Button-1>', self.read_move)
            self.game = renju.Game()
            self._player = self._black
            self.game_loop()


    def next_player(self):
        if self._player is self._black:
            self._player = self._white
        else:
            self._player = self._black


    def game_loop(self):
        pos = self._player.get_pos(self.game)

        if pos:
            assert self.game.is_possible_move(pos)
            self.game.move(pos)
            self.board_canvas.draw_stone(*pos, 
                                         color=str(self._player.color()), 
                                         move_n=self.game.move_n())

            if self._player.is_human():
                self._player.pos = None

            if not self.game:
                if self.game._result == renju.Player.WHITE:
                    winner = self._white.name() + ' [white]'
                else:
                    winner = self._black.name() + ' [black]'
                    
                self.bottom['text'] = f'{winner} win. Press (H) to play again.'
                self.master.bind('<KeyPress>', self.reset_game)
                self.after_cancel(self.alarm_id)                

                return self.game._result

            self.next_player()
            bottom_text = self._player.name() + ' turn now...'
            self.bottom['text'] = bottom_text

            if self._player.is_human():
                self.board_canvas.bind('<Button-1>', self.read_move)

        self.alarm_id = self.after(100, self.game_loop)


    def test_loop(self):
        while True:
            res = self.game_loop()
            if res == renju.Player.BLACK or res == renju.Player.WHITE:
                return res


    def read_move(self, event):
        'Process human move on mouse click'
        for i in range(15):
            for j in range(15):
                pixel_x = (i + 2) * 30
                pixel_y = (j + 2) * 30
                square_x = (event.x - pixel_x)**2
                square_y = (event.y - pixel_y)**2
                distance = math.sqrt(square_x + square_y)

                if distance < 15 and self.game.is_possible_move((i, j)):
                    # save position and unbind until next player finishes
                    # his move
                    self._player.pos = (i, j)
                    self.board_canvas.unbind('<Button-1>')
                    return





class BoardCanvas(tkinter.Canvas):
    def __init__(self, master=None, height=0, width=0):
        tkinter.Canvas.__init__(self, master, height=height, width=width)
        
        # init
        self.width = width
        self.height = height
        self.info = None
        self.legend_font = tkinter.font.Font(family='Monaco', size=14, weight='normal')
        self.move_font = tkinter.font.Font(family='Monaco', size=10, weight='bold')
        self.draw_board()


    def draw_board(self):
        # draw background
        self.create_rectangle(45, 45, 495, 495, fill='#f2e2c9', outline='#f2e2c9')

        LETTER = 'abcdefghjklmnop'
        # 15 horizontal lines
        for i in range(17):
            start_pixel_x = (i + 1) * 30
            start_pixel_y = (0 + 1) * 30
            end_pixel_x = (i + 1) * 30
            end_pixel_y = (16 + 1) * 30
            self.create_line(start_pixel_x, start_pixel_y, end_pixel_x, end_pixel_y, fill='#71767c')
            
            if 0 < i < 16:
                self.add_text(start_pixel_x, end_pixel_y + 15, f'{LETTER[i-1]}', font=self.legend_font)

        # 15 vertical lines
        for j in range(17):
            start_pixel_x = (0 + 1) * 30
            start_pixel_y = (j + 1) * 30
            end_pixel_x = (16 + 1) * 30
            end_pixel_y = (j + 1) * 30
            self.create_line(start_pixel_x, start_pixel_y, end_pixel_x, end_pixel_y, fill='#71767c')

            if 0 < j < 16:
                self.add_text(start_pixel_x - 15, start_pixel_y, f'{16 - j}', font=self.legend_font)

        
    def add_text(self, x, y, text, **kwargs):
        text_id = self.create_text(x, y, text=text, **kwargs)
        

    def display_info(self, text):
        if self.info:
            self.delete(self.info)
        self.info = self.create_text(self.width / 2, self.height - 20, text=text, font=self.legend_font)



    def draw_stone(self, row, col, color, move_n):
        start_pixel_x = (row + 2) * 30 - 12
        start_pixel_y = (col + 2) * 30 - 12
        end_pixel_x = (row + 2) * 30 + 12
        end_pixel_y = (col + 2) * 30 + 12

        self.create_oval(start_pixel_x, start_pixel_y, end_pixel_x, end_pixel_y, fill=color)
        text_color = 'black' if color is 'white' else 'white'
        self.add_text((row + 2) * 30, (col + 2) * 30, f'{move_n}', font=self.move_font, fill=text_color)
            

def run_gui(black, white):
    game = renju.Game()
    root = tkinter.Tk()
    app = GomokuUI(master=root, black=black, white=white, timeout=0)
    root.title("Gomoku")
    root.mainloop()

def run_test(black, white, games_count=100):
    black_win = 0
    white_win = 0

    

    for i in range(games_count):
        game = renju.Game()
        cur = black
        while game:
            game.move(cur.get_pos(game))
            if cur is black:
                cur = white
            else:
                cur = black

        if game.result() == renju.Player.BLACK:
            black_win += 1
        else:
            white_win += 1

        print(f'STEP {i}, white={white_win} black={black_win}')

    white_win /= games_count
    black_win /= games_count

    print(f'black -- {black_win}\nwhite -- {white_win}')

if __name__ == '__main__':
    #run_test(agent.SLAgent(modelfile='models/model.augmentations.03.hdf5', color=renju.Player.BLACK),
    #         agent.SLAgent(modelfile='models/model03.hdf5', color=renju.Player.WHITE))
    
    #run_test(agent.DummyAgent(color=renju.Player.BLACK),
    #         agent.DummyAgent(color=renju.Player.WHITE),
    #         games_count = 1000)

    run_gui(agent.HumanAgent(color=renju.Player.BLACK),
            agent.SLAgent(modelfile='models/model.augmentations.01.hdf5', color=renju.Player.WHITE))

    #run_test(agent.SLAgent(modelfile='models/model03.hdf5', color=renju.Player.BLACK),
    #         agent.TreeAgent(modelfile='models/model03.hdf5', max_depth=10, num_iters=500, color=renju.Player.WHITE))

    #run_test(agent.DummyAgent(color=renju.Player.BLACK),
    #         agent.DummyAgent(color=renju.Player.WHITE), games_count=10000)

