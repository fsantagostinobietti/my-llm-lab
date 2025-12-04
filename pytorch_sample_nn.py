# see https://apxml.com/posts/pytorch-macos-metal-gpu

import torch
from torch import nn
import time

print("pytorch version:", torch.__version__)
accelerator_available = torch.accelerator.is_available()
print(f"Accelerator available: {accelerator_available}")

# Check device
device = torch.accelerator.current_accelerator() if accelerator_available else torch.device("cpu")
device_name = device.type.upper()
print(f"Using accelerator device: {device_name}")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1+9*9, 512),
            nn.ReLU(),
            nn.Linear(512, 9),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
    
model = NeuralNetwork().to(device)
print(model)

SIZE_SAMPLES = (1+9*9,)
#SIZE_SAMPLES = (2, 1+9*9)
dim_samples = len(SIZE_SAMPLES) - 1
X = torch.rand(SIZE_SAMPLES, device=device)
print(X.size())
print("X:", X)

logits = model(X)
print("Logits:", logits)
pred_probab = nn.Softmax(dim=dim_samples)(logits)
print("Predicted probabilities:", pred_probab)
y_pred = pred_probab.argmax(dim=dim_samples)
print(f"Predicted class: {y_pred}")