import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import EnhancedCNNModel
from tqdm import tqdm
import numpy as np
import argparse
import os

# Constants (UPPER_CASE naming convention)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(dataset_path="./data"):
    """Load and prepare CIFAR-10 dataset with enhanced augmentations."""
    # Train transforms with more aggressive augmentations
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    # Test transforms (no augmentation)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    # Load datasets
    trainset = torchvision.datasets.CIFAR10(
        root=dataset_path, train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root=dataset_path, train=False, download=True, transform=transform_test
    )
    
    # Use pinned memory if CUDA is available for faster data transfer
    pin_memory = DEVICE.type == "cuda"
    trainloader = DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=4, pin_memory=pin_memory
    )
    testloader = DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=4, pin_memory=pin_memory
    )
    
    return trainloader, testloader

def train_epoch(model, trainloader, criterion, optimizer, epoch):
    """Train the model for one epoch with progress tracking."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
    for inputs, targets in pbar:
        inputs, targets = inputs.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)  # Slightly faster than zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        pbar.set_postfix({
            'loss': running_loss / (pbar.n + 1),
            'acc': 100. * correct / total
        })
    
    return running_loss / len(trainloader), 100. * correct / total

def evaluate(model, testloader, criterion):
    """Evaluate the model on the test set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(testloader, desc="Testing", unit="batch")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1),
                'acc': 100. * correct / total
            })
    
    return running_loss / len(testloader), 100. * correct / total

def main(args):
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    trainloader, testloader = load_data(args.data_dir)
    
    # Initialize model, loss, and optimizer
    model = EnhancedCNNModel().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Training loop
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, epoch)
        test_loss, test_acc = evaluate(model, testloader, criterion)
        scheduler.step()
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
        
        print(f"Epoch {epoch+1}: "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
    
    print(f"\nBest Test Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline CIFAR-10 Training")
    parser.add_argument('--data_dir', type=str, default="./data", help="Path to dataset")
    parser.add_argument('--output_dir', type=str, default="./output", help="Path to save models")
    args = parser.parse_args()
    
    main(args)