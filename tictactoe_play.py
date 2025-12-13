from enum import StrEnum
import numpy as np
import torch

from tictactoe import TicTacToe
from tictactoe_nn_1 import TicTacToeNeuralNetwork_1, from_game_to_one_hot


class GameResult(StrEnum):
    A_WINS = 'A'
    B_WINS = 'B'
    DRAW   = 'X'

class TicTacToeGame:
    """Tic Tac Toe game state and logic"""
    def __init__(self):
        self.moves = ""
        self.current_player = 'A'  # 'A' for first player, 'B' for second
        self.game_over = False
        self.result = None
        self.user_is_A = True  # True if user is first player (X), False if second (O)
        # Load the trained neural network model
        self.model = TicTacToeNeuralNetwork_1()
        try:
            checkpoint = torch.load("ttt_nn_1.pth", map_location=torch.device('cpu'))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
        except FileNotFoundError:
            print("Warning: Checkpoint file 'ttt_nn_1.pth' not found. Computer will use random moves.")
            self.model = None

        
    def make_move(self, position: int) -> bool:
        """Make a move at the given position (1-9). Returns True if successful."""
        if self.game_over or str(position) in self.moves:
            return False
        self.moves += str(position)
        self.result = TicTacToe.game_result(self.moves)
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
        """Computer makes a move using the neural network if available, otherwise random. 
           Returns the move made or None if no moves."""
        valid = self.get_valid_moves()
        if not valid:
            return None
        
        if self.model is not None:
            # Use neural network to predict best move
            input_str = self.moves.ljust(9, '0')
            input_tensor = from_game_to_one_hot(input_str).unsqueeze(0)  # add batch dimension
            with torch.no_grad():
                logits = self.model(input_tensor)
                probs = torch.softmax(logits, dim=1).squeeze()
            # Get probabilities for valid positions (indices 0-8)
            valid_indices = [int(p) - 1 for p in valid]
            valid_probs = probs[valid_indices]
            # Choose the move with highest probability
            best_idx = torch.argmax(valid_probs)
            chosen_pos = valid[best_idx]
        else:
            # Fallback to random move
            chosen_pos = int(np.random.choice(valid))
        
        self.make_move(chosen_pos)
        return chosen_pos

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
            print("You are O (second player), Computer is X (first player)")
        else: # user starts
            self.user_is_A = True
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
    game = TicTacToeGame()
    game.play_game()
        