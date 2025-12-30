
import torch


def from_move_to_one_hot(move: str) -> torch.Tensor:
    """Convert a move string to a one-hot encoded tensor."""
    one_hot = torch.zeros(9)
    move_index = int(move) - 1  # Convert move '1'-'9' to index 0-8
    if 0 <= move_index < 9:
        one_hot[move_index] = 1
    return one_hot

def from_game_to_one_hot(game: str) -> torch.Tensor:
    """Convert a game string to a one-hot encoded tensor."""
    moves = game[:]  # e.g. '357600000'

    # Encode moves
    return torch.cat( tuple(from_move_to_one_hot(move) for move in moves) )