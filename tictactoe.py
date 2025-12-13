from enum import StrEnum
import numpy as np


class GameResult(StrEnum):
    A_WINS = 'A'
    B_WINS = 'B'
    DRAW   = 'X'

class TicTacToe:
    """Tic Tac Toe game logic"""
    
    @staticmethod
    def game_result(moves: str) -> GameResult | None:
        """Determine the result of the game given the moves string."""
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
            game_result = TicTacToe.game_result(permutation[:n])
            if game_result is not None:
                return permutation[:n] + game_result
        return None # should not reach here
        
