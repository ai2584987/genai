import io
from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image

from helper_lib.generator import generate_grid
from helper_lib.model import get_model

WEIGHTS_PATH = Path(__file__).resolve().parent / "models" / "gan_mnist_generator.pth"


class DigitGenerator:
    def __init__(self, weights_path=WEIGHTS_PATH):
        weights_path = Path(weights_path)
        if not weights_path.is_file():
            raise FileNotFoundError(
                f"GAN weights not found at {weights_path}. "
                "Run notebooks/03_gan_mnist.ipynb to train and save them."
            )
        self.model = get_model("GAN")
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.model.generator.load_state_dict(state)  # only the generator is needed to make images
        self.model.eval()

    def generate_png(self, num_samples=16, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        grid = generate_grid(self.model, "cpu", num_samples)  # 3 x H x W in [0, 1]
        array = (grid.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8")
        image = Image.fromarray(array).resize(
            (grid.shape[2] * 4, grid.shape[1] * 4), Image.NEAREST  # 4x larger so digits are visible
        )
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()


@lru_cache(maxsize=1)
def get_digit_generator():
    return DigitGenerator()
