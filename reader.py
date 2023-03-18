import numpy as np
import pickle
import random
import torch
import chess.pgn
import sys
import copy
import random
from torch.utils.data import Dataset, DataLoader

'''
this thing shows up in predictor ipynb and it turns train.pgn into what we see as black/white_wins.txt
for some reason does not work in local 
but since txt files already given i guess safe to skip this part
archived here just in case
'''
def parsing():
    f1 = open('white_wins.txt', 'w')
    f2 = open('black_wins.txt', 'w')
    pgn = open("train.pgn")
    n_games = 0

    while True:
        game = chess.pgn.read_game(pgn)
        print(game)
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



'''
reading the txt files of winning cases and actually do some pre-proc
0 experience pre-proc anything so shitty code
'''
def readtxt():
    white_win_raw = np.loadtxt('predictor/white_wins.txt', delimiter=' ', skiprows=0, dtype=str)
    black_win_raw = np.loadtxt('predictor/black_wins.txt', delimiter=' ', skiprows=0, dtype=str)
    #print(white_win_raw[530],white_win_raw[531][0],white_win_raw[532])
    # now white/black_win_raw is a 2d array of chars
    # e.g. white_win_data[0] = ['r' 'n' 'b' 'q' 'k' 'b' 'n' 'r']
    # e.g. while_win_data[0][0] = 'r'
    # need more processing
    print("raw data loaded")
    white_win_games, black_win_games = [],[]
    row_number = -1
    game = []
    while row_number < len(white_win_raw)-1:
        
        if white_win_raw[row_number][0] == '-' and white_win_raw[row_number+1][0] == '=':
            white_win_games.append(copy.deepcopy(game))
            game = []
            row_number+=1 # now pointing at ======== line and ready for next game
        else:
            board = []
            for i in range(8):
                row_number+=1
                board.append(white_win_raw[row_number])
            game.append(board)
            row_number+=1 #row num pointing at -------- line at the end of each board
    #print(white_win_games[0][0][0][0])

    row_number = -1
    game = []
    while row_number < len(black_win_raw)-1:
        if black_win_raw[row_number][0] == '-' and black_win_raw[row_number+1][0] == '=':
            black_win_games.append(copy.deepcopy(game))
            game = []
            row_number+=1 # now pointing at ======== line and ready for next game
        else:
            board = []
            for i in range(8):
                row_number+=1
                board.append(black_win_raw[row_number])
            game.append(board)
            row_number+=1 #row num pointing at -------- line at the end of each board
    #print(black_win_games[0][0][0])
    print("game data parsed")
    # now black/white_win_games is list of games where in each game there are multiple game boards (8*8) aligned in the sequence they were played
    # doing further proc
    sets = [[[],[]],[[],[]],[[],[]],[[],[]],[[],[]],[[],[]],[[],[]],[[],[]],[[],[]]]
    #testing = []
    white_count = 0
    for game in white_win_games:
        for board in game:
            #board is a 8*8 array of char
            #need to make a separate board for each piece
            datum = []
            pawn = np.full((8,8),0)
            bishop = np.full((8,8),0)
            knight = np.full((8,8),0)
            rook = np.full((8,8),0)
            queen = np.full((8,8),0)
            king = np.full((8,8),0)
            for i in range(8):
                for j in range(8):
                    if board[i][j] == 'p': 
                        pawn[i][j] = -1
                    elif board[i][j] == 'P':
                        pawn[i][j] = 1
                    elif board[i][j] == 'b':
                        bishop[i][j] = -2
                    elif board[i][j] == 'B':
                        bishop[i][j] = 2
                    elif board[i][j] == 'n':
                        knight[i][j] = -3
                    elif board[i][j] == 'N':
                        knight[i][j] = 3
                    elif board[i][j] == 'r':
                        rook[i][j] = -4
                    elif board[i][j] == 'R':
                        rook[i][j] = 4
                    elif board[i][j] == 'q':
                        queen[i][j] = -5
                    elif board[i][j] == 'Q':
                        queen[i][j] = 5
                    elif board[i][j] == 'k':
                        king[i][j] = -6
                    elif board[i][j] == 'K':
                        king[i][j] = 6
            datum.append(pawn)
            datum.append(bishop)
            datum.append(knight)
            datum.append(rook)
            datum.append(queen)
            datum.append(king)
            datum = make_tensor(datum)
            t = (1,0)
            
            #just rolling a random dice to put things into 10 roughly even sets
            dice = random.randint(0,9)
            sets[dice][0].append(datum)
            sets[dice][1].append(t)
            white_count += 1
            if white_count % 10000 == 0:
                print("processed white boards:", white_count)
            
    black_count = 0
    for game in black_win_games:
        for board in game:
            #board is a 8*8 array of char
            #need to make a separate board for each piece
            datum = [] #datum = [whole, pawn, bishop, knight, rook, queen, king, white_win, black_win]
            pawn = np.full((8,8),0)
            bishop = np.full((8,8),0)
            knight = np.full((8,8),0)
            rook = np.full((8,8),0)
            queen = np.full((8,8),0)
            king = np.full((8,8),0)
            for i in range(8):
                for j in range(8):
                    if board[i][j] == 'p': 
                        pawn[i][j] = -1
                    elif board[i][j] == 'P':
                        pawn[i][j] = 1
                    elif board[i][j] == 'b':
                        bishop[i][j] = -2
                    elif board[i][j] == 'B':
                        bishop[i][j] = 2
                    elif board[i][j] == 'n':
                        knight[i][j] = -3
                    elif board[i][j] == 'N':
                        knight[i][j] = 3
                    elif board[i][j] == 'r':
                        rook[i][j] = -4
                    elif board[i][j] == 'R':
                        rook[i][j] = 4
                    elif board[i][j] == 'q':
                        queen[i][j] = -5
                    elif board[i][j] == 'Q':
                        queen[i][j] = 5
                    elif board[i][j] == 'k':
                        king[i][j] = -6
                    elif board[i][j] == 'K':
                        king[i][j] = 6
                    
            datum.append(pawn)
            datum.append(bishop)
            datum.append(knight)
            datum.append(rook)
            datum.append(queen)
            datum.append(king)
            datum = make_tensor(datum)
            t = (0,1)
            
            dice = random.randint(0,9)
            sets[dice][0].append(datum)
            sets[dice][1].append(t)
            black_count += 1
            if black_count%10000==0:
                print("processed black boards:", black_count)
    

    print("training sets made")
    '''
    for example, a datum that contains a king's gambit opening of a game in which the white player eventually won can look like this:
    
    set[i][j]=
    pair(
    [
        [
            ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'],
            ['p', 'p', 'p', 'p', '.', 'p', 'p', 'p'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', 'p', '.', '.', '.'],
            ['.', '.', '.', '.', 'P', 'P', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['P', 'P', 'P', 'P', '.', '.', 'P', 'P'],
            ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']
        ],
        [
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', 'p', '.', '.', '.'],
            ['.', '.', '.', '.', 'P', 'P', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['P', 'P', 'P', 'P', '.', '.', 'P', 'P'],
            ['.', '.', '.', '.', '.', '.', '.', '.']
        ],
        [
            ['.', '.', 'b', '.', '.', 'b', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', 'B', '.', '.', 'B', '.', '.']
        ],
        [
            ['.', 'n', '.', '.', '.', '.', 'n', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', 'N', '.', '.', '.', '.', 'N', '.']
        ], 
        [
            ['r', '.', '.', '.', '.', '.', '.', 'r'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['R', '.', '.', '.', '.', '.', '.', 'R']
        ], 
        [
            ['.', '.', '.', 'q', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', 'Q', '.', '.', '.', '.']
        ], 
        [
            ['.', '.', '.', '.', 'k', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', 'K', '.', '.', '.']
        ],
    ]
    ,
    tuple(1,0)
    )

    '''

    #print(training[0])
    # now both training and testing are filled with datum which are boards broken down to each kind of pieces plus a pair of annotation on the game's ending
    # each datum is 8*8*7 + 2, where the 8*8*7 would be fed to training and last 2 the reference/supervision provided for learning
    return white_win_games,black_win_games,sets

def make_tensor(datum):
    ts=torch.ones((6,8,8))
    #print(datum)
    #print(ts)
    for i in range(6):
        for j in range(8):
            for k in range(8):
                try:
                    ts[i][j][k] = datum[i][j][k]
                except:
                    print(datum, ts)
                    continue
    return ts

def main():
    #white,black,sets = readtxt()
    return None
main()