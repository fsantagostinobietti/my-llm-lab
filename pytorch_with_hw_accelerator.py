# see https://apxml.com/posts/pytorch-macos-metal-gpu

import torch
print("pytorch version:", torch.__version__)
mps_available = torch.backends.mps.is_available()
print(f"MPS available: {mps_available}")

# Check device
device = torch.device("mps" if mps_available else "cpu")
print(f"Using device: {str(device).upper()}")

# Example: Tensor operations
x = torch.rand(3, 3).to(device)
y = torch.rand(3, 3).to(device)
z = x + y
print(z)

# Example: Neural network
model = torch.nn.Linear(3, 1).to(device)
input_tensor = torch.rand(1, 3).to(device)
output = model(input_tensor)
print(output)


import time

x = torch.rand(1000, 1000)

# CPU Benchmark
start = time.time()
for _ in range(100):
    y = x @ x
end = time.time()
print(f"CPU time: {end - start:.4f} sec")

# MPS Benchmark
x = x.to("mps")
start = time.time()
for _ in range(100):
    y = x @ x
end = time.time()
print(f"MPS time: {end - start:.4f} sec")