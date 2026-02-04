import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from data_utils import ImagenetteDataModule
from models import BasicCNN, AllConvNet
import torch
import os

def train_model(model_class, name, dm):
    print(f"\n--- Training {name} ---")
    model = model_class()
    
    # Logger
    logger = CSVLogger("logs", name=name)
    
    # Callbacks
    early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=5)
    checkpoint = ModelCheckpoint(monitor="val_loss", mode="min", dirpath=f"checkpoints/{name}", filename="best")
    
    # Trainer
    trainer = L.Trainer(
        max_epochs=50,
        callbacks=[early_stop, checkpoint],
        logger=logger,
        accelerator="auto",
        devices=1
    )
    
    trainer.fit(model, datamodule= dm)
    results = trainer.test(model, datamodule=dm, ckpt_path="best")
    return results

def main():
    # Setup Data
    dm = ImagenetteDataModule()
    dm.prepare_data()
    dm.setup()
    
    # Task 1: Basic CNN
    basic_results = train_model(BasicCNN, "BasicCNN", dm)
    
    # Task 2: All Convolutional Net
    conv_results = train_model(AllConvNet, "AllConvNet", dm)
    
    # Parameter Comparison
    basic_model = BasicCNN()
    conv_model = AllConvNet()
    basic_params = sum(p.numel() for p in basic_model.parameters())
    conv_params = sum(p.numel() for p in conv_model.parameters())
    
    print("\n--- Model Comparison ---")
    print(f"Basic CNN Total Parameters: {basic_params:,}")
    print(f"AllConvNet Total Parameters: {conv_params:,}")
    print(f"Basic CNN Test Accuracy: {basic_results[0]['test_accuracy']:.4f}")
    print(f"AllConvNet Test Accuracy: {conv_results[0]['test_accuracy']:.4f}")

if __name__ == "__main__":
    main()
