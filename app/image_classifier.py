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
WEIGHTS_PATH = Path(__file__).resolve().parent / "models" / "cnn_cifar10.pth"


class ImageClassifier:
    def __init__(self, weights_path=WEIGHTS_PATH):
        weights_path = Path(weights_path)
        if not weights_path.is_file():
            raise FileNotFoundError(
                f"Model weights not found at {weights_path}. "
                "Run notebooks/02_cnn_cifar10.ipynb to train and save them."
            )
        self.model = get_model("CNN")
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        # Accept either a bare state_dict or a full checkpoint from helper_lib.checkpoints
        self.model.load_state_dict(state.get("model_state_dict", state))
        self.model.eval()
        self.transform = get_transform(64)  # same preprocessing as training

    def predict(self, image_bytes):
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
            raise ValueError("Could not read the file as an image") from error

        x = self.transform(image).unsqueeze(0)  # 1 x 3 x 64 x 64
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)[0]

        idx = int(torch.argmax(probs))
        return {"predicted_class": CLASSES[idx], "confidence": round(float(probs[idx]), 4)}


@lru_cache(maxsize=1)
def get_classifier():
    """Load the model once, on first use."""
    return ImageClassifier()
