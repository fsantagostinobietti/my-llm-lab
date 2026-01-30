import torch
from torch import nn


#
# Transformer v2 - added input embedding 
#                  positional embedding - it needs to fix context size; no improvement noticed
#                  "position-wise" FFN
#                  residual connections - no improvement noticed
#                  dropout - no improvement noticed
#                  layer normalization - no improvement noticed
#
CTX_SZ = 9 # context size (actually max game length size)
VOCAB_SZ = 10 # dictionary size (0-9)
D_IN = 9 # internal embedding size
NN_LAYER_MULTIPLIER = 4  # determines feed-forward layer size
NUM_BLOCKS = 3  # number of transformer blocks

def causal_mask(S: torch.Tensor) -> torch.Tensor:
    """Apply causal mask to attention scores"""
    mask = torch.triu(torch.ones(S.shape), diagonal=1)
    return S.masked_fill(mask.bool(), -torch.inf)

class LayerNorm(nn.Module):
    """Implementation of layer normalization that operates on the last dimension of
       the input tensor x, which represents the embedding dimension (emb_dim).

       The variable 'eps' is a small constant (epsilon) added to the variance to prevent division by zero
       during normalization. 
       The 'scale' and 'shift' are trainable parameters adjusted during training."""
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

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
        A = torch.softmax(S, dim=-1)  # attention weights for input X
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
        self.norm1 = LayerNorm(D_IN)
        self.norm2 = LayerNorm(D_IN)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.norm1(X)
        Z = X + self.attention(X) # attention with residual connection
        Z = self.norm2(Z)
        Z = Z + self.linear_relu_stack(Z) # apply FFN to each embedding
        return Z
    
class TicTacToeLM_1(nn.Module):
    """Input: moves one hot encoded (9x9 inputs).

    Uses a simple self-attention mechanism followed by feed-forward layers."""
    def __init__(self):
        super().__init__()
        # input embedding parameters
        self.Wemb = nn.Embedding(num_embeddings=VOCAB_SZ, embedding_dim=D_IN)
        # positional embedding layer
        self.pos_emb = torch.nn.Embedding(CTX_SZ, D_IN)
        # transfomer blocks
        self.trf_blocks = nn.Sequential(*[TTTTransformer() for _ in range(NUM_BLOCKS)])
        
    def forward(self, x: torch.Tensor):
        #print("Input x shape:", x.shape)
        X = self.Wemb(x) # embedding encoding (*batch_sz, CTX_SZ, D_IN)
        X = X + self.pos_emb(torch.arange(CTX_SZ))  # add positional embedding (using tensor broadcasting)
        Z :torch.Tensor = self.trf_blocks(X)
        logits = Z @ self.Wemb.weight.t()  # embedding decoding (use tight weight sharing)
        return logits
    
    def parameters_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

if __name__ == "__main__":
    model = TicTacToeLM_1()
    total_params = model.parameters_count()
    print(f"Total parameters: {total_params}")

