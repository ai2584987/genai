import io
from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError

from helper_lib.data_loader import get_transform
from helper_lib.model import get_model

# Same order as torchvision's CIFAR10().classes
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

# Resolved from this file, so it works whatever folder the server is started from
DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parent / "models" / "cnn_cifar10.pth"


class ImageClassifier:
    def __init__(self, weights_path=DEFAULT_WEIGHTS_PATH):
        weights_path = Path(weights_path)
        if not weights_path.is_file():
            raise FileNotFoundError(
                f"Model weights not found at {weights_path}. "
                "Run Assignment2__CNN.ipynb to train the model and create them."
            )
        self.model = get_model("CNN")
        state = torch.load(weights_path, map_location="cpu", weights_only=True)  
        self.model.load_state_dict(state.get("model_state_dict", state))
        self.model.eval()
        # Same preprocessing as training (single source of truth in helper_lib)
        self.transform = get_transform(64)

    def predict(self, image_bytes):
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")  # handles PNG/JPG, drops alpha
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
            raise ValueError("Could not read the data as an image") from error

        x = self.transform(image).unsqueeze(0)  # add batch dim -> 1x3x64x64
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)[0]

        idx = int(torch.argmax(probs))
        return {
            "predicted_class": CLASSES[idx],
            "confidence": round(float(probs[idx]), 4),
        }


@lru_cache(maxsize=1)
def get_classifier():
    """Load the model once, on the first request, instead of when the server imports the module."""
    return ImageClassifier()
