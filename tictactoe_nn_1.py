# see https://apxml.com/posts/pytorch-macos-metal-gpu


from torch import nn


#
# NeuralNetwork definition
#
class TicTacToeNeuralNetwork_1(nn.Module):
    """Input: moves one hot encoded (9x9 inputs)"""
    def __init__(self, layer_sz: int):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(9*9, layer_sz),
            nn.ReLU(),
            nn.Linear(layer_sz, 9),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
    
