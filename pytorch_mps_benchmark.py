
import torch, time

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
cpu = torch.device("cpu")

def bench_transfer(n=50, shape=(1024, 1024)):
    # Create CPU tensor once, measure repeated transfers to MPS
    x_cpu = torch.randn(*shape, device=cpu)
    if device.type != "mps":
        print("MPS not available; skipping.")
        return
    # Warm-up
    for _ in range(10):
        _ = x_cpu.to(device)
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        _ = x_cpu.to(device)
    torch.mps.synchronize()
    print("Avg transfer CPU -> MPS:", (time.perf_counter() - t0)/n, "s")

def bench_compute(n=50, shape=(1024, 1024)):
    x_cpu = torch.randn(*shape, device=cpu)
    x_mps = torch.randn(*shape, device=device)
    # Warm-up
    for _ in range(10):
        _ = x_mps @ x_mps.T
    if device.type == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        _ = x_mps @ x_mps.T
    if device.type == "mps":
        torch.mps.synchronize()
    print("Avg matmul on MPS:", (time.perf_counter() - t0)/n, "s")
    # CPU baseline
    for _ in range(10):
        _ = x_cpu @ x_cpu.T
    t0 = time.perf_counter()
    for _ in range(n):
        _ = x_cpu @ x_cpu.T
    print("Avg matmul on CPU:", (time.perf_counter() - t0)/n, "s")

bench_transfer()
bench_compute()
