import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader 
from torchvision import datasets, transforms, models


def get_teacher_model():
    pretrained_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(512, 10)
    model_state_dict = pretrained_model.state_dict()

    for key in list(model_state_dict.keys()):
        if key not in model.state_dict() or model_state_dict[key].shape != model.state_dict()[key].shape:
            del model_state_dict[key]

    model.load_state_dict(model_state_dict, strict=False)
    return model


def train_teacher(model, trainloader, criterion, optimizer, epochs, device):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        total = 0
        correct = 0
        for imgs, labels in trainloader:
            imgs, labels = imgs.to(device), labels.to(device)   
            optimizer.zero_grad(set_to_none=True)

            outs = model(imgs)
            loss = criterion(outs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outs, 1)   
            total += labels.size(0)
            correct += (predicted == labels).sum().item()   
        
        epoch_loss = running_loss / len(trainloader)    
        epoch_accuracy = 100 * correct / total
        print(f"epoch: {epoch + 1}, loss: {epoch_loss:.4f}, accuracy: {epoch_accuracy:.2f}%")


def evaluate(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in testloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outs = model(imgs)
            _, predicted = torch.max(outs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"test accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    batch_size = 64 
    learning_rate = 0.01    
    epochs = 5

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)  
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)    

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = get_teacher_model().to(device)

    criterion = nn.CrossEntropyLoss()   
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_teacher(model, trainloader, criterion, optimizer, epochs, device)
    evaluate(model, testloader, device)

    torch.save(model, "trained_teacher_1.pth")

