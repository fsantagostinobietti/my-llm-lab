from enum import StrEnum
import numpy as np


class GameResult(StrEnum):
    A_WINS = 'A'
    B_WINS = 'B'
    DRAW = 'X'

class TicTacToe:
    """Tic Tac Toe game state and logic"""
    def __init__(self):
        self.moves = ""
        self.current_player = 'A'  # 'A' for first player, 'B' for second
        self.game_over = False
        self.result = None
        self.user_is_A = True  # True if user is first player (X), False if second (O)

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
        
    def make_move(self, position: int) -> bool:
        """Make a move at the given position (1-9). Returns True if successful."""
        if self.game_over or str(position) in self.moves:
            return False
        self.moves += str(position)
        self.result = self._game_result(self.moves)
        if self.result:
            self.game_over = True
        else:
            self.current_player = 'B' if self.current_player == 'A' else 'A'
        return True

    def get_valid_moves(self) -> list[int]:
        """Return list of valid moves (1-9)."""
        all_positions = set('123456789')
        used = set(self.moves)
        return sorted([int(p) for p in all_positions - used])

    def computer_move(self) -> int | None:
        """Computer makes a random valid move. Returns the move made or None if no moves."""
        valid = self.get_valid_moves()
        if valid:
            move = int(np.random.choice(valid))
            self.make_move(move)
            return move
        return None

    def display_board(self):
        """Display the current board state."""
        board = [' ']*9
        for i, move in enumerate(self.moves):
            pos = int(move) - 1
            board[pos] = 'X' if i % 2 == 0 else 'O'
        
        print(f" {board[0]} | {board[1]} | {board[2]} ")
        print("---+---+---")
        print(f" {board[3]} | {board[4]} | {board[5]} ")
        print("---+---+---")
        print(f" {board[6]} | {board[7]} | {board[8]} ")
        print()

    def play_game(self):
        """Start the interactive game loop."""
        choice = input("Do you want to go first? (y/n): ").lower().strip()
        if choice == 'n': # computer starts
            self.user_is_A = False
            #self.current_player = 'A'
            print("You are O (second player), Computer is X (first player)")
        else: # user starts
            self.user_is_A = True
            #self.current_player = 'A'
            print("You are X (first player), Computer is O (second player)")
        
        while not self.game_over:
            self.display_board()
            is_user_turn = (self.current_player == 'A' and self.user_is_A) or (self.current_player == 'B' and not self.user_is_A)
            if is_user_turn:
                try:
                    move = int(input("Your move (1-9): "))
                    if move < 1 or move > 9:
                        print("Invalid move. Choose 1-9.")
                        continue
                    if not self.make_move(move):
                        print("Invalid move. Position taken.")
                        continue
                except ValueError:
                    print("Invalid input. Enter a number 1-9.")
                    continue
            else:  # computer
                print("Computer's turn...")
                move = self.computer_move()
                print(f"Computer plays {move}")
        self.display_board()
        if self.result == GameResult.DRAW:
            print("It's a draw!")
        elif (self.result == GameResult.A_WINS and self.user_is_A) or (self.result == GameResult.B_WINS and not self.user_is_A):
            print("You win!")
        else:
            print("Computer wins!")
            


if __name__ == "__main__":
    game = TicTacToe()
    game.play_game()
        