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


# ---------------------------------------------------------------------------
# Assignment 3: GAN for MNIST (28x28 grayscale digits)
# ---------------------------------------------------------------------------
Z_DIM = 100


class Generator(nn.Module):
    """
    Input:  B x 100 noise vector
    Linear(100 -> 7*7*128), reshape                          -> B x 128 x 7 x 7
    ConvTranspose2d(128 -> 64, k4, s2, p1) + BatchNorm + ReLU -> B x 64 x 14 x 14
    ConvTranspose2d(64 -> 1,  k4, s2, p1) + Tanh              -> B x 1 x 28 x 28
    """

    def __init__(self, z_dim=Z_DIM):
        super().__init__()
        self.z_dim = z_dim
        self.fc = nn.Linear(z_dim, 7 * 7 * 128)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.deconv2 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, z):
        x = self.fc(z).view(-1, 128, 7, 7)
        x = F.relu(self.bn1(self.deconv1(x)))
        return torch.tanh(self.deconv2(x))  # pixels in [-1, 1]


class Discriminator(nn.Module):
    """
    Input:  B x 1 x 28 x 28
    Conv2d(1 -> 64,   k4, s2, p1) + LeakyReLU(0.2)             -> B x 64 x 14 x 14
    Conv2d(64 -> 128, k4, s2, p1) + BatchNorm + LeakyReLU(0.2) -> B x 128 x 7 x 7
    Flatten + Linear(128*7*7 -> 1) + Sigmoid                    -> B x 1 (probability "real")
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 7 * 7, 1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        return torch.sigmoid(self.fc(self.flatten(x)))


class GAN(nn.Module):
    """Holds the generator and discriminator together so get_model('GAN') returns one object."""

    def __init__(self, z_dim=Z_DIM):
        super().__init__()
        self.z_dim = z_dim
        self.generator = Generator(z_dim)
        self.discriminator = Discriminator()

    def forward(self, z):
        return self.generator(z)


def get_model(model_name):
    """Return the model for model_name: one of FCNN, CNN, EnhancedCNN, GAN."""
    models = {"FCNN": MLP, "CNN": SimpleCNN, "EnhancedCNN": EnhancedCNN, "GAN": GAN}
    if model_name not in models:
        raise ValueError(f"Unknown model_name '{model_name}'. Choose one of: {', '.join(models)}")
    return models[model_name]()
