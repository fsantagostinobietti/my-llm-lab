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
    def __init__(self, game_generator):
        super().__init__()
        self.game_generator = game_generator

    def _extract_training_sample(self, game: str) -> tuple[str, str]:
        """Extract a training sample from a game string.
        
        Args:
            game: Game string with moves followed by result (e.g. '357624B')
            
        Returns:
            Tuple of (input_moves, output_move) for training
        """
        game_result = game[-1]  # 'A', 'B', or 'X'
        moves = game[:-1]       # e.g. '357624'
        
        # Determine player and valid move indices
        if game_result in ['A', 'B']:
            player = game_result
        else:  # game_result == 'X'
            player = random.choice(['A', 'B'])
        # A plays on even indices, B on odd
        valid_indices = range(1 if player == 'B' else 0, len(moves), 2)
        
        # Select a random move for this player
        idx = random.choice(list(valid_indices))
        input_moves = player + moves[:idx].ljust(9, '0')  # pad to length 9 with '0's
        output_move = moves[idx]
        
        return (input_moves, output_move)

    @staticmethod
    def _from_move_to_one_hot(move: str) -> torch.Tensor:
        """Convert a move string to a one-hot encoded tensor."""
        one_hot = torch.zeros(9)
        move_index = int(move) - 1  # Convert move '1'-'9' to index 0-8
        if 0 <= move_index < 9:
            one_hot[move_index] = 1
        return one_hot

    @staticmethod
    def _from_game_to_one_hot(game: str) -> torch.Tensor:
        """Convert a game string to a one-hot encoded tensor."""
        player = game[0]  # 'A' or 'B'
        moves = game[1:]  # e.g. '357600000'
        
        # Encode player
        player_bit = torch.zeros(1) 
        player_bit[0] = 0 if player == 'A' else 1

        # Encode moves
        one_hot_sequence = torch.cat( tuple(TicTacToeStreamDataset._from_move_to_one_hot(move) for move in moves) )

        return torch.cat((player_bit, one_hot_sequence))
    
    def __iter__(self):
        """Returns an iterator of input moves vs predicted next move.
        Input is prefixed with player we want to predict for.
        E.g. 'A357600000' -> '2'"""
        while True:
            game: str = self.game_generator()
            print("Generated game:", game)
            inputs, output = self._extract_training_sample(game)
            yield self._from_game_to_one_hot(inputs), self._from_move_to_one_hot(output)

        
if __name__ == '__main__':
    TicTacToeIterable = TicTacToeStreamDataset(TicTacToe.generate_random_game)

    # DataLoader on IterableDataset does not support shuffling
    dataloader = DataLoader(dataset=TicTacToeIterable, batch_size=3, num_workers=0)
    for batch in dataloader:
        print(batch[0].shape, batch[1].shape)
        break

