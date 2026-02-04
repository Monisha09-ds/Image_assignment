import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from data_utils import ImagenetteDataModule
from models import BasicCNN
import torch

def main():
    print("\n--- Training Regularized BasicCNN ---")
    
    # 1. Setup Data with Augmentation
    dm = ImagenetteDataModule()
    dm.prepare_data()
    dm.setup()
    
    # Override train transform with augmented version for Task 3
    dm.train_ds.dataset.transform = dm.augmented_transform
    
    # 2. Setup Model with Dropout
    model = BasicCNN(use_dropout=True)
    
    # 3. Logger & Callbacks
    logger = CSVLogger("logs", name="RegularizedCNN")
    early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=5)
    checkpoint = ModelCheckpoint(monitor="val_loss", mode="min", dirpath="checkpoints/RegularizedCNN", filename="best")
    
    # 4. Trainer
    trainer = L.Trainer(
        # max_epochs=50,
        max_epochs=5,
        callbacks=[early_stop, checkpoint],
        logger=logger,
        accelerator="auto",
        devices=1
    )
    
    trainer.fit(model, datamodule=dm)
    results = trainer.test(model, datamodule=dm, ckpt_path="best")
    
    print("\n--- Regularization Comparison ---")
    print(f"Regularized Model Test Accuracy: {results[0]['test_accuracy']:.4f}")

if __name__ == "__main__":
    main()
