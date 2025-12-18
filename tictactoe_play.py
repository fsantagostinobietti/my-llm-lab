from enum import StrEnum
import random
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
    def __init__(self, model, checkpoint=None, device: torch.device = torch.device('cpu')):
        self.model = model
        if checkpoint is not None:
            try:
                checkpoint_data = torch.load(checkpoint, map_location=device)
                self.model.load_state_dict(checkpoint_data['model_state_dict'])
                self.model.eval()
            except FileNotFoundError:
                print(f"Warning: Checkpoint file '{checkpoint}' not found. AI will use random moves.")
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

class PerfectPlayer(Player):
    """Perfect player that uses minimax to never lose and maximize wins."""
    
    # def __init__(self, player: str):
    #     self.player = player  # 'A' or 'B'
    
    def next_move(self, game_moves: str) -> int:
        valid = self.get_valid_moves(game_moves)
        if not valid:
            return None
        
        # Compute scores for each valid move using minimax
        player = 'A' if len(game_moves) % 2 == 0 else 'B'
        scores = []
        for move in valid:
            new_moves = game_moves + str(move)
            score = self.minimax(new_moves, player)
            scores.append((move, score))
        
        # Find the maximum score
        max_score = max(s for m, s in scores)
        
        # Choose the first move with the maximum score
        for move, score in scores:
            if score == max_score:
                return move
    
    @staticmethod
    def minimax(moves, maximizer):
        result = TicTacToe.game_result(moves)
        if result:
            if result == GameResult.DRAW:
                return 0
            elif result == GameResult.A_WINS:
                return 1 if maximizer == 'A' else -1
            elif result == GameResult.B_WINS:
                return 1 if maximizer == 'B' else -1
        
        valid_moves = Player.get_valid_moves(moves)
        if not valid_moves:
            return 0
        
        scores = []
        for move in valid_moves:
            new_moves = moves + str(move)
            score = PerfectPlayer.minimax(new_moves, maximizer)
            scores.append(score)
        
        current_player = 'A' if len(moves) % 2 == 0 else 'B'
        if current_player == maximizer:
            return max(scores)
        else:
            return min(scores)

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

    def play_game(self, player_A: Player, player_B: Player, print_output: bool = True) -> GameResult:
        """Start the game loop with the given players."""
        self.current_player = 'A'
        while not self.game_over:
            if print_output:
                self.display_board()
            if self.current_player == 'A':
                move = player_A.next_move(self.moves)
            else:
                move = player_B.next_move(self.moves)
            if move is not None:
                if print_output:
                    # Print move for non-human players
                    if (self.current_player == 'A' and not isinstance(player_A, HumanPlayer)) or \
                    (self.current_player == 'B' and not isinstance(player_B, HumanPlayer)):
                        print(f"Player {self.current_player} plays {move}")
                self.make_move(move)
        if print_output:
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
    # Example: Perfect player vs Random player
    playerA = HumanPlayer()
    playerB = PerfectPlayer()
    game.play_game(playerA, playerB)

    # # Example: Human vs AI
    # player1 = HumanPlayer()
    # player2 = AIPlayer(checkpoint="ttt_nn_1.pth")
    # game.play_game(player1, player2)
    
    # # Example: Random player vs AI player
    # playerA = RandomPlayer()
    # playerB = AIPlayer(model=TicTacToeNeuralNetwork_1(layer_sz=512), checkpoint="ttt_nn_#ttt_nn_1.pth")
    # stats: dict[str, set] = {GameResult.A_WINS: set(), GameResult.B_WINS: set(), GameResult.DRAW: set()}
    # for _ in range(10000):
    #     game = TicTacToeGame()
    #     result = game.play_game(playerA, playerB, print_output=False)
    #     stats[result].add(game.moves)
    # print("Stats:")
    # for key, value in stats.items():
    #     print(f"  {key}: {len(value)}")
