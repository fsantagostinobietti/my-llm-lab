import torch
from torch import nn


#
# Transformer v2 - added input embedding 
#                  #positional embedding
#                  "position-wise" FFN
#
CTX_SZ = 9 # context size (actually max game length size)
FEAT_SZ = 9 # input feature size (actually one-hot single move encoding size)
D_IN = 9 #6 # internal embedding size
NN_LAYER_MULTIPLIER = 4  # determines feed-forward layer size
NUM_BLOCKS = 3  # number of transformer blocks

def causal_mask(S: torch.Tensor) -> torch.Tensor:
    """Apply causal mask to attention scores"""
    mask = torch.triu(torch.ones(S.shape), diagonal=1)
    return S.masked_fill(mask.bool(), -torch.inf)

class TTTAttention(nn.Module):
    """Self-attention mechanism for Tic Tac Toe"""
    def __init__(self):
        super().__init__()
        #self.W = torch.nn.Parameter(torch.randn(size=(D_IN, D_IN))) # weight parameter 
        self.Wq = torch.nn.Parameter(torch.randn(size=(D_IN, D_IN))) # query weights 
        self.Wk   = torch.nn.Parameter(torch.randn(size=(D_IN, D_IN))) # key weights 
        #self.dropout = nn.Dropout(0.1)
        #self.Wv = torch.nn.Parameter(torch.randn(size=(D_IN, D_IN))) # value weights (not used)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        #print("Input X shape:", X.shape)  # input size (CTX_SZ, D_IN)
        Q = X @ self.Wq  # queries
        K = X @ self.Wk  # keys
        S = (Q @ K.transpose(-2, -1)) / (D_IN ** 0.5)  # attention scores for input X
        S = causal_mask(S)
        A = torch.softmax(S, dim=-2)  # attention weights for input X
        #A = self.dropout(A) # optional: apply dropout to attention weights
        Z = A @ X # attention output
        return Z

class TTTTransformer(nn.Module):
    """Single Transformer block for Tic Tac Toe"""
    def __init__(self):
        super().__init__()
        # self-attention layer
        self.attention = TTTAttention()
        # feed forward layer
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(D_IN, D_IN * NN_LAYER_MULTIPLIER),
            nn.ReLU(),
            nn.Linear(D_IN * NN_LAYER_MULTIPLIER, D_IN),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        Z = self.attention(X)
        Z = self.linear_relu_stack(Z) # apply FFN to each embedding
        return Z
    
class TicTacToeLM_1(nn.Module):
    """Input: moves one hot encoded (9x9 inputs).

    Uses a simple self-attention mechanism followed by feed-forward layers."""
    def __init__(self):
        super().__init__()
        # input embedding parameters
        self.Wemb = torch.nn.Parameter(torch.randn(size=(FEAT_SZ, D_IN)))
        # # positional embedding layer
        # self.pos_emb = torch.nn.Embedding(CTX_SZ, D_IN)
        # transfomer blocks
        self.trf_blocks = nn.Sequential(*[TTTTransformer() for _ in range(NUM_BLOCKS)])
        
    def forward(self, x: torch.Tensor):
        #print("Input x shape:", x.shape)
        batch_sz = x.shape[:-1]
        X = x.reshape(*batch_sz, CTX_SZ, FEAT_SZ)  # input in matrix shape: CTX_SZ x FEAT_SZ
        X = X @ self.Wemb # embedding encoding
        #X = X + self.pos_emb(torch.arange(CTX_SZ))  # add positional embedding (using tensor broadcasting)
        Z :torch.Tensor = self.trf_blocks(X)
        # get last embedding only
        last_emb = Z[..., -1, :]  # shape: (*batch_sz, D_IN)
        logits = last_emb @ self.Wemb.transpose(-2, -1)  # embedding decoding
        return logits
    
    def parameters_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

if __name__ == "__main__":
    # test
    model = TicTacToeLM_1()
    total_params = model.parameters_count()
    print(f"Total parameters: {total_params}")

    x = torch.randn(size=(CTX_SZ*FEAT_SZ,))  # random input
    logits_single: torch.Tensor = model(x)

    x = x.unsqueeze(0)  # add batch dimension
    logits_batched: torch.Tensor = model(x)
    if torch.allclose(logits_single, logits_batched.squeeze(0)):
        print("Single and batch logits match!")
    else:
        print("Mismatch between single and batch logits!")
        print("Logits:", logits_single)
        print("Logits with batch dim:", logits_batched)
