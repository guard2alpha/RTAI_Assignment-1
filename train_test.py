#MNIST는 쉬운 모델이기에 학습하고 CIFAR10은 pre-trained model을 사용, + 정규화

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # Prevent OpenMP error

import torch
import torch.nn as nn
import torch.optim as optim

# Import custom modules
from datasets import get_dataloaders
from model import MNIST_CNN, get_cifar10_model

def train_and_evaluate_mnist(model, train_loader, test_loader, device, epochs=3):
    """
    Trains the MNIST model and evaluates its accuracy.
    Saves the weights to avoid retraining.
    """
    save_path = "mnist_cnn.pth"
    criterion = nn.CrossEntropyLoss()
    
    # Load pre-trained weights if they exist
    # if os.path.exists(save_path):
    #     print("Loading existing MNIST model weights...")
    #     model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    # else:
    print("Starting MNIST training...")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        total = 0
        correct = 0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Accuracy: {100 * correct/total:.4f}")
                
    print("Training complete! Saving model...")
    torch.save(model.state_dict(), save_path)

    # Evaluate accuracy (Target: >= 95%)
    print("Evaluating MNIST model accuracy on clean data...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    print(f"MNIST Clean Test Accuracy: {accuracy:.2f}%")
    return model

def evaluate_cifar10(model, test_loader, device):
    """
    Evaluates the pre-trained CIFAR-10 model accuracy. (Target: >= 80%)
    """
    print("Evaluating CIFAR-10 model accuracy on clean data...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    print(f"CIFAR-10 Clean Test Accuracy: {accuracy:.2f}%\n")
    return accuracy

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # 1. Load data
    mnist_train_loader, mnist_test_loader, cifar_test_loader = get_dataloaders()
    
    # 2. Setup, train, and evaluate MNIST model
    mnist_model = MNIST_CNN().to(device)
    mnist_model = train_and_evaluate_mnist(mnist_model, mnist_train_loader, mnist_test_loader, device, epochs=10)
    
    # 3. Load and evaluate CIFAR-10 model
    cifar_model = get_cifar10_model().to(device)
    evaluate_cifar10(cifar_model, cifar_test_loader, device)
    
    print("DONE")