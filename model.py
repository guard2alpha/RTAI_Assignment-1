import torch
import torch.nn as nn
import torchvision.transforms as transforms

# Normalization실시
class CIFAR10_Wrapper(nn.Module):
    """
    Wrapper that takes [0, 1] inputs and applies CIFAR-10 normalization 
    internally. This is crucial for adversarial attacks.
    """
    def __init__(self, base_model):
        super(CIFAR10_Wrapper, self).__init__()
        self.base_model = base_model
        
        # Official CIFAR-10 mean and std
        self.normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )

    def forward(self, x):
        # 1. Normalize [0, 1] input
        x = self.normalize(x)
        # 2. Pass the normalized image to the base model
        return self.base_model(x)
    
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
    Loads a pre-trained ResNet20 model for CIFAR-10 with an internal normalizer.
    Citation: https://github.com/chenyaofo/pytorch-cifar-models
    """
    print("Loading pre-trained CIFAR-10 ResNet20 model...")
    
    # Load base model
    base_model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
    
    # Wrap the base model
    model = CIFAR10_Wrapper(base_model)
    model.eval()
    
    return model