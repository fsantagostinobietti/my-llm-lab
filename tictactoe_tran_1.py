import torch
from torch import nn


#
# Transformer definition
#
CTX_SZ = 9 # context size (actually max game length size)
D_IN = 9 # input feature size (actually one-hot size per move)
D_OUT = 3  # embedding size

def causal_mask(S: torch.Tensor) -> torch.Tensor:
    """Apply causal mask to attention scores"""
    mask = torch.triu(torch.ones(CTX_SZ, CTX_SZ), diagonal=1)
    return S.masked_fill(mask.bool(), -torch.inf)

class TicaTacToeAttention(nn.Module):
    """Self-attention mechanism for Tic Tac Toe"""
    def __init__(self):
        super().__init__()
        #self.W = torch.nn.Parameter(torch.randn(size=(D_IN, D_IN))) # weight parameter (d_in x d_in)
        self.Wq = torch.nn.Parameter(torch.randn(size=(D_IN, D_OUT))) # query weights (d_in x d_out)
        self.Wk   = torch.nn.Parameter(torch.randn(size=(D_IN, D_OUT))) # key weights (d_in x d_out)
        #self.dropout = nn.Dropout(0.1)
        #self.W_value = torch.nn.Parameter(torch.randn(size=(D_IN, D_OUT))) # weight parameter

    def forward(self, x: torch.Tensor):
        #print("Input x shape:", x.shape)  # shape (CTX_SZ x D_IN)
        batch_sz = x.shape[:-1]
        X = x.reshape(*batch_sz, CTX_SZ, D_IN)  # attention input
        Q = X @ self.Wq  # queries
        K = X @ self.Wk  # keys
        S = (Q @ K.transpose(-2, -1)) / (D_IN ** 0.5)  # attention scores for input X
        S = causal_mask(S)
        A = torch.softmax(S, dim=-2)  # attention weights for input X
        #A = self.dropout(A) # optional: apply dropout to attention weights
        #print("A:", A)
        Z = A @ X # attention output
        z = Z.reshape(*batch_sz, CTX_SZ*D_IN)  # flatten
        return z

class TicTacToeTransformer_1(nn.Module):
    """Input: moves one hot encoded (9x9 inputs).

    Uses a simple self-attention mechanism followed by feed-forward layers.
    
    NB: W_query and W_key matrices are merged into the same square W for simplicity.
      (See https://github.com/rasbt/LLMs-from-scratch/discussions/517)"""
    def __init__(self, layer_sz: int):
        super().__init__()
        # attention parameters
        self.attention = TicaTacToeAttention()
        # feed forward parameters
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(CTX_SZ*D_IN, layer_sz),
            nn.ReLU(),
            nn.Linear(layer_sz, D_IN),
        )

    def forward(self, x: torch.Tensor):
        #print("Input x shape:", x.shape)
        z = self.attention(x)
        logits = self.linear_relu_stack(z)
        return logits
    
if __name__ == "__main__":
    # test
    model = TicTacToeTransformer_1(layer_sz=4)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    x = torch.randn(size=(CTX_SZ*D_IN,))  # random input
    logits_single: torch.Tensor = model(x)

    x = x.unsqueeze(0)  # add batch dimension
    logits_batched: torch.Tensor = model(x)
    if torch.allclose(logits_single, logits_batched.squeeze(0)):
        print("Single and batch logits match!")
    else:
        print("Mismatch between single and batch logits!")
        print("Logits:", logits_single)
        print("Logits with batch dim:", logits_batched)
