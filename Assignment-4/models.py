import torch
from torch import nn
import torch.nn.functional as F
import lightning as L
import torchmetrics

class BaseClassifier(L.LightningModule):
    def __init__(self, num_classes=10, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.lr = lr

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = self.accuracy(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accuracy", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = self.accuracy(y_hat, y)
        self.log("test_loss", loss)
        self.log("test_accuracy", acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

class BasicCNN(BaseClassifier):
    def __init__(self, num_classes=10, lr=1e-3, use_dropout=False):
        super().__init__(num_classes, lr)
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Input size: 1x64x64 -> 32x32x32 -> 64x16x16 -> 128x8x8
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5) if use_dropout else nn.Identity(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

class AllConvNet(BaseClassifier):
    """
    Simplified All-Convolutional Net based on Springenberg et al.
    No pooling layers. Strided convolutions used for downsampling.
    """
    def __init__(self, num_classes=10, lr=1e-3):
        super().__init__(num_classes, lr)
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), # 64x64
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=2), # 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 32x32
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2), # 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 16x16
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2), # 8x8
            nn.ReLU(),
            nn.Conv2d(128, num_classes, kernel_size=1), # 8x8 class scores
            nn.AdaptiveAvgPool2d(1), # Global Average Pooling
            nn.Flatten()
        )

    def forward(self, x):
        return self.model(x)
