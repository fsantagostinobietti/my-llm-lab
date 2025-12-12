from torch import nn
import torch
from torch.utils.data import DataLoader

from pytorch_dataloader import TicTacToe, TicTacToeStreamDataset
from pytorch_sample_nn import NeuralNetwork


learning_rate = 1e-3
batch_size = 64
epochs = 3
epoch_size = 100000

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
            if current >= epoch_size:
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
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

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

if __name__ == '__main__':
    accelerator_available = torch.accelerator.is_available()
    #device = torch.accelerator.current_accelerator() if accelerator_available else torch.device("cpu")
    device = torch.device("cpu")

    TicTacToeIterable = TicTacToeStreamDataset(TicTacToe.generate_random_game, device=device)

    # DataLoader on IterableDataset 
    train_dataloader = DataLoader(dataset=TicTacToeIterable, batch_size=batch_size, num_workers=0)
    test_dataloader = DataLoader(dataset=TicTacToeIterable, batch_size=batch_size)

    # Init modele and optimizer
    model = NeuralNetwork().to(device)
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters())

    # load checkpoint (if available)
    checkpoint_path = "checkpoint.pth"
    epoch = load_checkpoint(model, optimizer, checkpoint_path, device)

    loss_fn = nn.CrossEntropyLoss()

    for _ in range(epochs):
        epoch += 1
        print(f"Epoch {epoch}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")

    # Save checkpoint for later resume
    save_checkpoint(model, optimizer, epoch, checkpoint_path)