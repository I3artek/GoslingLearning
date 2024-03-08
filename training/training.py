import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os
from PIL import Image
import numpy as np

from torchvision.models import ResNet18_Weights
from torchvision.models import ResNet50_Weights


from torch import nn
import math
import torch
import torch.nn.functional as F

def main():
    # Set random seed for reproducibility
    #torch.manual_seed(42)

    # Define data transformations
    transform = transforms.Compose([
        transforms.Resize((206, 206)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

    # Load training and validation datasets
    train_path = "/content/Train_augprep"
    val_path = "/content/Validation_augprep"

    train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_path, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)

    # Define the ResNet18 model
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)

    num_classes = len(train_dataset.classes)

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 90)
    )

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    # Move the model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Check if a checkpoint exists and load it
    checkpoint_path = '/content/drive/MyDrive/ML/checkpoints/resnet18_checkpointxxxx.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Checkpoint loaded. Resuming training from epoch {epoch + 1}.")

    # Training loop
    num_epochs = 10000
    checkpoint_interval = 1 # Set the interval for saving checkpoints
    lowest_loss = 1000
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch + 1}")
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
            avg_val_loss = val_loss / len(val_loader)

            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Validation Loss: {avg_val_loss:.4f}')

        # Save the model checkpoint every 100 epochs
        if (epoch + 1) % checkpoint_interval == 0:
        #if avg_val_loss < lowest_loss:
            lowest_loss = avg_val_loss
            checkpoint_filename = f'/content/drive/MyDrive/ML/checkpoints_attempt2/experiments/resnet18_epoch_{epoch + 1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, checkpoint_filename)
            print(f'Model checkpoint saved: {checkpoint_filename}')


if __name__ == "__main__":
    main()
