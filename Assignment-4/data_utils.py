import torch
from torchvision import transforms, datasets
import lightning as L
from torch.utils.data import DataLoader, random_split

class ImagenetteDataModule(L.LightningDataModule):
    def __init__(self, data_dir="./data/imagenette", batch_size=128, img_size=64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size
        
        self.transform = transforms.Compose([
            transforms.CenterCrop(160),
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            transforms.Grayscale() 
        ])
        
        # Regularization (Data Augmentation) for Task 3
        self.augmented_transform = transforms.Compose([
            transforms.RandomResizedCrop(self.img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            transforms.Grayscale()
        ])

    def prepare_data(self):
        datasets.Imagenette(self.data_dir, split="train", size="160px", download=True)
        datasets.Imagenette(self.data_dir, split="val", size="160px", download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            full_train = datasets.Imagenette(self.data_dir, split="train", size="160px", transform=self.transform)
            train_size = int(len(full_train) * 0.9)
            val_size = len(full_train) - train_size
            self.train_ds, self.val_ds = random_split(full_train, [train_size, val_size], 
                                                     generator=torch.Generator().manual_seed(42))
        if stage == "test" or stage is None:
            self.test_ds = datasets.Imagenette(self.data_dir, split="val", size="160px", transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=4)

class CIFAR10DataModule(L.LightningDataModule):
    def __init__(self, data_dir="./data/cifar10", batch_size=128):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        
        self.transform = transforms.Compose([
            transforms.Resize(64), # Resize to match Imagenette models if needed
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            transforms.Grayscale()
        ])

    def prepare_data(self):
        datasets.CIFAR10(self.data_dir, train=True, download=True)
        datasets.CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            full_train = datasets.CIFAR10(self.data_dir, train=True, transform=self.transform)
            train_size = int(len(full_train) * 0.9)
            val_size = len(full_train) - train_size
            self.train_ds, self.val_ds = random_split(full_train, [train_size, val_size],
                                                     generator=torch.Generator().manual_seed(42))
        if stage == "test" or stage is None:
            self.test_ds = datasets.CIFAR10(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=4)
