import torch
import torchvision
import torchvision.transforms as transforms


def get_dataloaders(batch_size=64, resize=(224, 224)):

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
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


