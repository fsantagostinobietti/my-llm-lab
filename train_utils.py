from torch import nn
import torch
from torch.utils.data import DataLoader

EPOCH_SIZE = 100000
FILTER_PRED = [0, 2, 4, 6] # only positions for player B

def train_loop(dataloader: DataLoader, model: nn.Module, loss_fn: nn.modules.loss._Loss, optimizer: torch.optim.Optimizer):
    # Set the model to training mode 
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        batch_size = X.shape[0]
        # Compute prediction
        pred = model(X)
        # Compute loss
        y_mod = y[..., FILTER_PRED]
        y_flat = y_mod.reshape(-1)  # flatten to (batch_size * steps,)
        pred_mod = pred[..., FILTER_PRED, :]
        pred_flat = pred_mod.reshape(-1, pred.size(-1))  # flatten to (batch_size * steps, num_classes)
        loss: torch.Tensor = loss_fn(pred_flat, y_flat)
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

    test_loss, correct, tot_pred = 0, 0, 0
    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            # compute loss
            y_mod = y[..., FILTER_PRED]
            y_flat = y_mod.reshape(-1)  # flatten to (batch_size * steps,)
            pred_mod = pred[..., FILTER_PRED, :]
            pred_flat = pred_mod.reshape(-1, pred.size(-1))  # flatten to (batch_size * steps, num_classes)
            loss: torch.Tensor = loss_fn(pred_flat, y_flat)
            test_loss += loss.item()
            # compute accuracy
            correct += (pred_flat.argmax(1) == y_flat).type(torch.float).sum().item()
            tot_pred += len(y_flat)
            break # only one batch

    correct /= tot_pred  # accuracy ratio
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
