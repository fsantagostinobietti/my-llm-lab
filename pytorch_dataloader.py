from enum import StrEnum
import random
import time
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data import DataLoader


class GameResult(StrEnum):
    A_WINS = 'A'
    B_WINS = 'B'
    DRAW = 'X'

class TicTacToe:
    """Tic Tac Toe game state and logic"""
    def __init__(self):
        pass  # Implementation

    @staticmethod
    def _game_result(moves: str) -> GameResult | None:
        WINNING_POSITIONS = [
            {1,2,3}, # top row
            {4,5,6}, # middle row
            {7,8,9}, # bottom row
            {1,4,7}, # left row
            {2,5,8}, # center row
            {3,6,9}, # right row
            {1,5,9}, # up-left to down-right diagonal
            {3,5,7}  # up-right to down-left diagonal
        ]

        # get A moves (odd moves only)
        A_moves = moves[0::2]
        A_positions = set(int(m) for m in A_moves)
        res = GameResult.A_WINS if any(wp <= A_positions for wp in WINNING_POSITIONS) else None
        if res:
            return res
        # get B moves (even moves only)
        B_moves = moves[1::2]
        B_positions = set(int(m) for m in B_moves)
        res = GameResult.B_WINS if any(wp <= B_positions for wp in WINNING_POSITIONS) else None
        if res:
            return res
        return GameResult.DRAW if len(moves) == 9 else None

    @staticmethod
    def generate_random_game() -> str:
        """generates a random valid Tic Tac Toe game string.
        E.g. '357624B'"""
        permutation = ''.join(np.random.choice(list('123456789'), size=9, replace=False))
        for n in range(5, 10):  # shortest game is 5 moves (A wins), longest is 9
            game_result = TicTacToe._game_result(permutation[:n])
            if game_result is not None:
                return permutation[:n] + game_result
        return None # should not reach here
        

class TicTacToeFileDataset(Dataset):
    """dataset froma a CSV file for Tic Tac Toe games"""
    def __init__(self, games_file, input_transform=None, target_transform=None):
        pass  # Implementation
    def __getitem__(self, idx):
        pass  # Implementation

class TicTacToeStreamDataset(IterableDataset):
    """dataset from a stream for Tic Tac Toe games"""
    def __init__(self, game_generator, input_transform=None, target_transform=None):
        self.game_generator = game_generator
        self.input_transform = input_transform
        self.target_transform = target_transform
        
    def __iter__(self):
        """returns an iterator of samples in this dataset"""
        while True:
            yield self.game_generator()
        

TicTacToeIterable = TicTacToeStreamDataset(TicTacToe.generate_random_game)
for _ in range(5):
    print(next(iter(TicTacToeIterable)))

dataloader = DataLoader(dataset=TicTacToeIterable, batch_size=3, num_workers=0)
for batch in dataloader:
    print(batch)
    break
