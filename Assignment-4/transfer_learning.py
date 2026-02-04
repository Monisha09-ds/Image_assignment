import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from data_utils import CIFAR10DataModule
from models import BasicCNN
import torch
import os

def train_cifar(model, name, dm):
    print(f"\n--- Training {name} on CIFAR10 ---")
    logger = CSVLogger("logs_cifar", name=name)
    checkpoint = ModelCheckpoint(monitor="val_loss", mode="min", dirpath=f"checkpoints/cifar_{name}", filename="best")
    
    trainer = L.Trainer(
        # max_epochs=20, # Shorter for demo/fine-tuning
        max_epochs=3,
        callbacks=[checkpoint],
        logger=logger,
        accelerator="auto",
        devices=1
    )
    
    trainer.fit(model, datamodule=dm)
    results = trainer.test(model, datamodule=dm, ckpt_path="best")
    return results

def main():
    # Setup CIFAR10 Data
    dm_cifar = CIFAR10DataModule()
    dm_cifar.prepare_data()
    dm_cifar.setup()
    
    # 1. Train from Scratch
    print("Initializing model from scratch...")
    scratch_model = BasicCNN()
    scratch_results = train_cifar(scratch_model, "ScratchCNN", dm_cifar)
    
    # 2. Transfer Learning (Fine-tuning)
    print("\nInitializing model with pre-trained weights from Imagenette...")
    pretrained_path = "checkpoints/BasicCNN/best.ckpt"
    
    if os.path.exists(pretrained_path):
        # Load weights
        finetune_model = BasicCNN.load_from_checkpoint(pretrained_path)
        # In this case, num_classes is 10 for both, so no need to replace the head
        # but if we wanted to freeze features:
        # for param in finetune_model.features.parameters():
        #     param.requires_grad = False
    else:
        print(f"Warning: Pre-trained checkpoint {pretrained_path} not found. Running with random weights.")
        finetune_model = BasicCNN()
        
    finetune_results = train_cifar(finetune_model, "TransferCNN", dm_cifar)
    
    print("\n--- Transfer Learning Comparison ---")
    print(f"Scratch Training Test Accuracy: {scratch_results[0]['test_accuracy']:.4f}")
    print(f"Fine-tuning Test Accuracy: {finetune_results[0]['test_accuracy']:.4f}")

if __name__ == "__main__":
    main()
