# see https://apxml.com/posts/pytorch-macos-metal-gpu

import random
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset

from nn_utils import from_game_to_one_hot, from_move_to_one_hot
from tictactoe_play import generate_game_against_perfect_player
from tictactoe_tran_1 import TicTacToeTransformer_1
from train_utils import load_checkpoint, save_checkpoint, test_loop, train_loop


#
# Dataset
#
class TicTacToeStreamDataset(IterableDataset):
    """dataset from a stream for Tic Tac Toe games"""
    def __init__(self, game_generator, device: torch.device):
        super().__init__()
        self.game_generator = game_generator
        self.device = device

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
        input_moves = moves[:idx].ljust(9, '0')  # pad to length 9 with '0's
        output_move = moves[idx]
        return (input_moves, output_move)

    def __iter__(self):
        """Returns an iterator of input moves vs predicted next move.
        Input is prefixed with player we want to predict for.
        E.g. 'A357600000' -> '2'"""
        while True:
            game: str = self.game_generator()
            #print("Generated game:", game)
            inputs, output = self._extract_training_sample(game)
            yield from_game_to_one_hot(inputs).to(self.device), from_move_to_one_hot(output).to(self.device)


def train_transformer_1(layer_sz: int, epochs: int, batch_size: int):
    #accelerator_available = torch.accelerator.is_available()
    #device = torch.accelerator.current_accelerator() if accelerator_available else torch.device("cpu")
    device = torch.device("cpu")

    #TicTacToeIterable = TicTacToeStreamDataset(TicTacToe.generate_random_game, device=device)
    TicTacToeIterable = TicTacToeStreamDataset(generate_game_against_perfect_player, device=device)
    

    # DataLoader on IterableDataset 
    train_dataloader = DataLoader(dataset=TicTacToeIterable, batch_size=batch_size, num_workers=0)
    test_dataloader = DataLoader(dataset=TicTacToeIterable, batch_size=batch_size)

    # Init modele and optimizer
    model = TicTacToeTransformer_1(layer_sz).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    # load checkpoint (if available)
    checkpoint_path = "ttt_tran_1.pth"
    epoch = load_checkpoint(model, optimizer, checkpoint_path, device)

    loss_fn = nn.CrossEntropyLoss()

    for _ in range(epochs):
        epoch += 1
        print(f"Epoch {epoch} ", end="")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")

    # Save checkpoint for later resume
    save_checkpoint(model, optimizer, epoch, checkpoint_path)


if __name__ == '__main__':
    # run training session
    train_transformer_1(layer_sz=4, epochs=300, batch_size=64)