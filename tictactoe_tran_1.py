import torch
from torch import nn


#
# Transformer definition
#
CTX_SZ = 9 # context size i.e. game board size
FEATURE_SZ = 9 # feature size i.e. one hot encoded move

class TicTacToeTranformer_1(nn.Module):
    """Input: moves one hot encoded (9x9 inputs)"""
    def __init__(self, layer_sz: int):
        super().__init__()
        # attention parameters
        self.W = torch.nn.Parameter(torch.randn(size=(FEATURE_SZ, FEATURE_SZ))) # weight parameter
        # feed forward parameters
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(CTX_SZ*FEATURE_SZ, layer_sz),
            nn.ReLU(),
            nn.Linear(layer_sz, FEATURE_SZ),
        )

    def forward(self, x: torch.Tensor):
        X = x.reshape(CTX_SZ, FEATURE_SZ)  # attention input
        A = torch.softmax(X @ self.W @ X.T, dim=1)  # attention weights
        Z = A @ X # attention output
        z = Z.reshape(-1)  # flatten
        logits = self.linear_relu_stack(z)
        return logits
    
if __name__ == "__main__":
    # test
    model = TicTacToeTranformer_1(layer_sz=8)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    x = torch.randn(size=(CTX_SZ*FEATURE_SZ,))  # random input
    logits = model(x)
    print("Logits:", logits)