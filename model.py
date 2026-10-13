import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 10


class MLP(nn.Module):
    """Fully connected baseline from Module 4 Practical 1 (FCNN)."""

    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 32 * 3, 200)
        self.fc2 = nn.Linear(200, 150)
        self.fc3 = nn.Linear(150, NUM_CLASSES)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=1)
        return x


class SimpleCNN(nn.Module):
    """
    CNN for 64x64 RGB images, built on the SimpleCNN from Module 4 Practical 3.

    Input:  B x 3 x 64 x 64
    Conv2d(3 -> 16, 3x3, stride 1, padding 1) + ReLU  -> B x 16 x 64 x 64
    MaxPool2d(2x2, stride 2)                          -> B x 16 x 32 x 32
    Conv2d(16 -> 32, 3x3, stride 1, padding 1) + ReLU -> B x 32 x 32 x 32
    MaxPool2d(2x2, stride 2)                          -> B x 32 x 16 x 16
    Flatten                                           -> B x 8192
    Linear(8192 -> 100) + ReLU                        -> B x 100
    Linear(100 -> 10)                                 -> B x 10
    """

    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 100)
        self.fc2 = nn.Linear(100, NUM_CLASSES)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class EnhancedCNN(nn.Module):
    """CNN with BatchNorm and Dropout for 64x64 input."""

    def __init__(self):
        super(EnhancedCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        # 64x64 downsampled 4 times -> 4x4 spatial dimensions
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        x = x.view(-1, 128 * 4 * 4)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def get_model(model_name):
    """Return the model for model_name: one of FCNN, CNN, EnhancedCNN."""
    if model_name == "FCNN":
        model = MLP()
    elif model_name == "CNN":
        model = SimpleCNN()
    elif model_name == "EnhancedCNN":
        model = EnhancedCNN()
    else:
        raise ValueError(
            f"Unknown model_name '{model_name}'. Choose one of: FCNN, CNN, EnhancedCNN"
        )
    return model
