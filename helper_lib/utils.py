import torch


def get_device():
    """Same device selection used in the module notebooks."""
    return (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )


def save_model(model, path='model.pth'):
    """Save only the model weights."""
    torch.save(model.state_dict(), path)
