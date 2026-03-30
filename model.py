import torch
import torch.nn as nn

class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        
        # 28x28 to 7x7 after double 2x2 poolings
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        
        x = x.view(-1, 32 * 7 * 7) # Flatten
        
        x = self.relu(self.fc1(x))
        x = self.fc2(x) # Return logits
        return x

def get_cifar10_model():
    """
    Loads a pre-trained ResNet20 model for CIFAR-10.
    Citation: https://github.com/chenyaofo/pytorch-cifar-models
    """
    print("Loading pre-trained CIFAR-10 ResNet20 model...")
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
    
    model.eval() 
    return model