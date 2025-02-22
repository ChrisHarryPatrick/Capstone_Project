# src/data_loader.py
import torch
import torchvision
import torchvision.transforms as transforms

def get_datasets():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # MNIST Dataset
    mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # CIFAR-10 Dataset
    cifar_train = torchvision.datasets.CIFAR10(root='./data', train=True, 
download=True, transform=transform)
    cifar_test = torchvision.datasets.CIFAR10(root='./data', train=False, 
download=True, transform=transform)

    return mnist_train, mnist_test, cifar_train, cifar_test

if __name__ == "__main__":
    mnist_train, mnist_test, cifar_train, cifar_test = get_datasets()
    print(f"MNIST Train: {len(mnist_train)}, MNIST Test: {len(mnist_test)}")
    print(f"CIFAR-10 Train: {len(cifar_train)}, CIFAR-10 Test: {len(cifar_test)}")

