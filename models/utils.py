import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image   

def get_dataloaders(batch_size=32, resize=(28, 28)):

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(resize, interpolation=Image.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    trainloader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True
    )

    testloader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform),
        batch_size=batch_size, shuffle=False
    )

    return trainloader, testloader  


