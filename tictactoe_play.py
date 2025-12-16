from enum import StrEnum
import random
import numpy as np
import torch

from tictactoe import TicTacToe
from tictactoe_nn_1 import TicTacToeNeuralNetwork_1, from_game_to_one_hot


class Player:
    """Base class for all players."""

    @staticmethod
    def get_valid_moves(moves: str) -> list[int]:
        """Return list of valid moves (1-9)."""
        all_positions = set('123456789')
        used = set(moves)
        return sorted([int(p) for p in all_positions - used])
    
    def next_move(self, game_moves: str) -> int:
        """Make a move in the game. Returns the position (1-9)."""
        raise NotImplementedError

class HumanPlayer(Player):
    """Human player that makes moves via user input."""
    def next_move(self, game_moves: str) -> int:
        while True:
            try:
                move = int(input("Your move (1-9): "))
                if move in self.get_valid_moves(game_moves):
                    return move
                else:
                    print("Invalid move. Position taken.")
            except ValueError:
                print("Invalid input. Enter a number 1-9.")

class RandomPlayer(Player):
    """Random player that makes random valid moves."""
    def next_move(self, game_moves: str) -> int:
        valid = self.get_valid_moves(game_moves)
        return random.choice(valid) if valid else None

class AIPlayer(Player):
    """AI player that makes moves using a neural network."""
    def __init__(self, model=None, checkpoint=None, device: torch.device = torch.device('cpu')):
        if model is not None:
            self.model = model
        elif checkpoint is not None:
            self.model = TicTacToeNeuralNetwork_1()
            try:
                checkpoint_data = torch.load(checkpoint, map_location=device)
                self.model.load_state_dict(checkpoint_data['model_state_dict'])
                self.model.eval()
            except FileNotFoundError:
                print(f"Warning: Checkpoint file '{checkpoint}' not found. AI will use random moves.")
                self.model = None
        else:
            self.model = None

    def next_move(self, game_moves: str) -> int:
        valid = self.get_valid_moves(game_moves)
        if not valid:
            return None
        
        if self.model is not None:
            # Use neural network to predict best move
            input_str = game_moves.ljust(9, '0')
            input_tensor = from_game_to_one_hot(input_str)
            with torch.no_grad():
                logits = self.model(input_tensor)
                probs = torch.softmax(logits, dim=0)
            # Get probabilities for valid positions
            valid_indices = [int(p) - 1 for p in valid]
            valid_probs = probs[valid_indices]
            # Normalize probabilities
            valid_probs = valid_probs / valid_probs.sum()
            # Choose the move with highest probability
            best_idx = torch.argmax(valid_probs)
            return valid[best_idx]
        else:
            # Fallback to random move
            return random.choice(valid)

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

    def display_board(self):
        """Display the current board state."""
        # board placeholders: ¹²³⁴⁵⁶⁷⁸⁹  ₁₂₃₄₅₆₇₈₉
        board = list("¹²³⁴⁵⁶⁷⁸⁹")
        for i, move in enumerate(self.moves):
            pos = int(move) - 1
            board[pos] = 'X' if i % 2 == 0 else 'O'
        
        print(f" {board[0]} | {board[1]} | {board[2]} ")
        print("---+---+---")
        print(f" {board[3]} | {board[4]} | {board[5]} ")
        print("---+---+---")
        print(f" {board[6]} | {board[7]} | {board[8]} ")
        print()

    def play_game(self, player_A: Player, player_B: Player) -> GameResult:
        """Start the game loop with the given players."""
        self.current_player = 'A'
        while not self.game_over:
            self.display_board()
            if self.current_player == 'A':
                move = player_A.next_move(self.moves)
            else:
                move = player_B.next_move(self.moves)
            if move is not None:
                # Print move for non-human players
                if (self.current_player == 'A' and not isinstance(player_A, HumanPlayer)) or \
                   (self.current_player == 'B' and not isinstance(player_B, HumanPlayer)):
                    print(f"Player {self.current_player} plays {move}")
                self.make_move(move)
        self.display_board()
        if self.result == GameResult.DRAW:
            print("It's a draw!")
        elif self.result == GameResult.A_WINS:
            print("Player A wins!")
        else:
            print("Player B wins!")
        return self.result
            


if __name__ == "__main__":
    game = TicTacToeGame()
    # # Example: Human vs AI
    # player1 = HumanPlayer()
    # player2 = AIPlayer(checkpoint="ttt_nn_1.pth")
    # game.play_game(player1, player2)
    
    # Example: Random vs AI
    playerA = RandomPlayer()
    playerB = AIPlayer(checkpoint="ttt_nn_1.pth")
    game.play_game(playerA, playerB)