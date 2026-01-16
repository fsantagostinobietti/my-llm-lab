# Training utility for Tic Tac Toe models

import random
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset

from nn_utils import from_game_to_one_hot, from_move_to_one_hot, to_token_ids
from tictactoe_nn_1 import TicTacToeNeuralNetwork_1
from tictactoe_play import generate_game_against_perfect_player
from tictactoe_tran_1 import TicTacToeTransformer_1
from tictactoe_lm_1 import TicTacToeLM_1
from train_utils import load_checkpoint, save_checkpoint, test_loop, train_loop


#
# Dataset
#
class TicTacToeStreamDataset(IterableDataset):
    """dataset from a stream for Tic Tac Toe games"""
    def __init__(self, game_generator, device: torch.device, game_selector: str | None = None):
        super().__init__()
        self.game_generator = game_generator
        self.device = device
        self.game_selector = game_selector

    def _extract_training_sample(self, game: str) -> tuple[str, str]:
        """Extract a training sample from a game string.
        
        Args:
            game: Game string with moves followed by result (e.g. '357624B')
            
        Returns:
            Tuple of (input_moves, output_move) for training
        """
        moves = game[:-1]       # e.g. '357624'
        input_moves = moves.ljust(9, '0')  # pad to length 9 with '0's
        output_move = moves[1:].ljust(9, '0')  # next move prediction, pad to length 9 with '0's
        return (input_moves, output_move)

    def __iter__(self):
        """Returns an iterator of input moves vs predicted next move.
        Input is prefixed with player we want to predict for.
        E.g. '357600000' -> '576200000'"""
        while True:
            game: str = self.game_generator()
            # skip games not matching 'game_selector'
            game_result = game[-1]  # 'A', 'B', or 'X'
            if self.game_selector and game_result not in list(self.game_selector):
                continue # skip this game
            #print("Generated game:", game)
            inputs, output = self._extract_training_sample(game)
            yield to_token_ids(inputs).to(self.device), to_token_ids(output).to(self.device)

def train_model(model: nn.Module, epochs: int, batch_size: int, checkpoint_path: str, player: str | None = None):
    #accelerator_available = torch.accelerator.is_available()
    #device = torch.accelerator.current_accelerator() if accelerator_available else torch.device("cpu")
    device = torch.device("cpu")

    game_selector = player+'X' if player in ['A', 'B'] else None # include draws if player specified
    TicTacToeIterable = TicTacToeStreamDataset(generate_game_against_perfect_player, device=device, game_selector=game_selector)
    
    # DataLoader on IterableDataset 
    train_dataloader = DataLoader(dataset=TicTacToeIterable, batch_size=batch_size, num_workers=0)
    test_dataloader = DataLoader(dataset=TicTacToeIterable, batch_size=batch_size)

    # Init modele and optimizer
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters()) #torch.optim.Adam(model.parameters())

    # load checkpoint (if available)
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

    # # run training session for TicTacToeNeuralNetwork_1
    # train_model(TicTacToeNeuralNetwork_1(layer_sz=4), epochs=3, batch_size=64, checkpoint_path="ttt_nn_1.pth")
    
    # # run training session for TicTacToeTransformer_1
    # train_model(TicTacToeTransformer_1(layer_sz=4), epochs=3, batch_size=64, checkpoint_path="ttt_tran_1.pth")

    # run training session for TicTacToeLM_1 for player B only
    train_model(TicTacToeLM_1(), epochs=300, batch_size=64, checkpoint_path="ttt_lm_1_B.pth", player="B")
