import numpy as np
import pickle
import random
import torch
import chess.pgn
import sys
from torch.utils.data import Dataset, DataLoader

def parsing():
    f1 = open('white_wins.txt', 'w')
    f2 = open('black_wins.txt', 'w')
    pgn = open("train.pgn")
    n_games = 0

    while True:
        game = chess.pgn.read_game(pgn)
        try:
            result = game.headers["Result"]
        except:
            break

        if result == "1-0":
            n_games =+ 1
            board = game.board()
            for move in game.main_line():
                board.push(move)
                f1.write(str(board))
                f1.write("\n- - - - - - - -\n")
            f1.write("\n= = = = = = = =\n")
        if result == "0-1":
            n_games =+ 1
            board = game.board()
            for move in game.main_line():
                board.push(move)
                f2.write(str(board))
                f2.write("\n- - - - - - - -\n")
            f2.write("\n= = = = = = = =\n")

    f1.close()        
    f2.close()
    pgn.close()
    print("Done. ", n_games, " games parsed")

