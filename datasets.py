# dataset.py
import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(batch_size=64):


    transform = transforms.Compose([transforms.ToTensor()])
    
    # 1. MNIST 
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    mnist_train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    mnist_test_loader = DataLoader(mnist_test, batch_size=1, shuffle=False)
    
    # 2. CIFAR-10
    cifar_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    cifar_test_loader = DataLoader(cifar_test, batch_size=1, shuffle=False) 
    
    print("downloaded")
    return mnist_train_loader, mnist_test_loader, cifar_test_loader

if __name__ == "__main__":
    get_dataloaders()