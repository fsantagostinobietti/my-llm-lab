# see https://apxml.com/posts/pytorch-macos-metal-gpu

import torch
import time

print("pytorch version:", torch.__version__)
accelerator_available = torch.accelerator.is_available()
print(f"Accelerator available: {accelerator_available}")

# Check device
device = torch.accelerator.current_accelerator() if accelerator_available else torch.device("cpu")
device_name = device.type.upper()
print(f"Using accelerator device: {device_name}")

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

if accelerator_available:
    # Benchmarking
    x = torch.rand(1000, 1000)

    # CPU Benchmark
    start = time.time()
    for _ in range(100):
        y = x @ x
    end = time.time()
    print(f"CPU time: {end - start:.4f} sec")

    # Accelerator Benchmark
    x = x.to(device)
    start = time.time()
    for _ in range(100):
        y = x @ x
    end = time.time()
    print(f"{device_name} time: {end - start:.4f} sec")