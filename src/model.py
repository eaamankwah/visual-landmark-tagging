import torch
import torch.nn as nn


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()

        # Convolutional feature extractor.
        # Input images are 224x224x3 (see src/data.py). Each block is
        # Conv -> BatchNorm -> ReLU -> MaxPool, which halves the spatial
        # dimensions and doubles the number of channels, a classic pattern
        # for CNNs from scratch (VGG-style).
        self.features = nn.Sequential(
            # 3 x 224 x 224 -> 16 x 112 x 112
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 16 x 112 x 112 -> 32 x 56 x 56
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 32 x 56 x 56 -> 64 x 28 x 28
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 64 x 28 x 28 -> 128 x 14 x 14
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 128 x 14 x 14 -> 256 x 7 x 7
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Classifier head. Dropout amount is controlled by the "dropout"
        # parameter, and the output size is controlled by "num_classes"
        # (never hardcoded), as required by the rubric.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Process the input tensor through the feature extractor, then
        # through the classifier. Note: we intentionally do NOT apply
        # softmax here -- nn.CrossEntropyLoss expects raw logits, and
        # softmax is applied later at inference time (see src/predictor.py)
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience method (used for the "stand out" image-retrieval
        suggestion): returns the penultimate-layer features (i.e., right
        before the final classification layer) for an input batch. These
        features can be used to compute similarity between images, e.g.
        via a dot product, in order to implement a simple image-retrieval
        system.
        """
        x = self.features(x)
        # Run through the classifier except for the very last Linear layer
        for layer in self.classifier[:-1]:
            x = layer(x)
        return x


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.__next__()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
