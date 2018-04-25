import itertools
import time
import tkinter as tk
from tkinter import font


import renju
import agent

class GomokuUI(tk.Frame):
    def __init__(self, master=None, black=None, white=None):
        super().__init__(master)
        self.create_widgets()
        self.black = black
        self.white = white
        self.game = renju.Game()
        # Both players are not humans
        if black and white:
            self.board_canvas.unbind('<Button-1>')
            while self.game:
                self.game_loop(None)
    
    def create_widgets(self):
        self.board_canvas = BoardCanvas(height=600, width=530)
        self.board_canvas.bind('<Button-1>', self.game_loop)
        self.board_canvas.pack()


    def game_loop(self, event):
        pass


    def update(self, game, probs):
        move_n = game.move_n()
        if move_n:
            pos = game.last_pos()
            color = str(game._player.another())
            self.board_canvas.draw_stone(pos[0], pos[1], color, move_n)

class BoardCanvas(tk.Canvas):
    def __init__(self, master=None, height=0, width=0):
        tk.Canvas.__init__(self, master, height=height, width=width)
        
        # init
        self.legend_font = font.Font(family='Monaco', size=14, weight='normal')
        self.move_font = font.Font(family='Monaco', size=10, weight='bold')
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
        text_id = self.create_text(x, y, **kwargs)
        self.itemconfigure(text_id, text=text)
        

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

    root = tk.Tk()
    app = GomokuUI(master=root)
    app.master.title("Gomoku")

    for game, probs in renju.loop(game, black, white):
        app.update(game, probs)
        time.sleep(.5)
        root.update_idletasks()
        root.update()
    
    app.update(game, probs)
    root.mainloop()

if __name__ == '__main__':
    run_gui(agent.DummyAgent('black'), agent.DummyAgent('white'))
