import torch
from torchvision.utils import make_grid


@torch.no_grad()
def generate_grid(model, device="cpu", num_samples=16, nrow=None):
    """Sample num_samples latent vectors, run the generator, return a 3 x H x W image grid in [0, 1]."""
    gen = model.generator.to(device).eval()
    z = torch.randn(num_samples, model.z_dim, device=device)
    images = gen(z).cpu()                 # values in [-1, 1]
    images = (images + 1) / 2             # back to [0, 1]
    nrow = nrow or int(num_samples ** 0.5) or 1
    return make_grid(images, nrow=nrow, padding=2, pad_value=1.0)


def generate_samples(model, device="cpu", num_samples=10):
    """Module 6 class activity: generate samples and plot them on a grid."""
    import matplotlib.pyplot as plt

    grid = generate_grid(model, device, num_samples)
    plt.figure(figsize=(6, 6))
    plt.imshow(grid.permute(1, 2, 0), cmap="gray")
    plt.axis("off")
    plt.show()
