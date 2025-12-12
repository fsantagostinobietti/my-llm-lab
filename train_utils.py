from torch import nn
import torch
from torch.utils.data import DataLoader


batch_size = 64
EPOCH_SIZE = 100000

def train_loop(dataloader: DataLoader, model: nn.Module, loss_fn: nn.modules.loss._Loss, optimizer: torch.optim.Optimizer):
    #size = len(dataloader.dataset)
    # Set the model to training mode 
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        #print(X, y)
        # Compute prediction and loss
        pred = model(X)
        #print("Predictions:", pred)
        loss: torch.Tensor = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            #print(f"loss: {loss:>7f}  [{current:>5d}]")
            print(".", end="")
            if current >= EPOCH_SIZE:
                print()
                break

def test_loop(dataloader: DataLoader, model: nn.Module, loss_fn: nn.modules.loss._Loss):
    # Set the model to evaluation mode
    model.eval()
    
    test_loss, correct = 0, 0
    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            #print(X, y)
            pred = model(X)
            #print("Predictions:", pred, "pred.argmax(1):", pred.argmax(1), "y.argmax(1):", y.argmax(1))
            loss: torch.Tensor = loss_fn(pred, y)
            test_loss += loss.item()
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
            break # only one batch

    correct /= batch_size
    print(f"Validation Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, path: str):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f"Checkpoint saved at epoch {epoch} to {path}")

def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, checkpoint_path: str, device: torch.device) -> int:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        print(f"Loaded checkpoint from epoch {epoch}")
    except FileNotFoundError:
        print("No checkpoint found, starting from scratch")
        epoch = 0
    return epoch
