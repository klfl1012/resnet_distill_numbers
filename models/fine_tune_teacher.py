import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from teacher import Teacher, TeacherCNN
from student import StudentCNN
from utils import get_dataloaders
import json


def train_teacher(model, trainloader, criterion, optimizer, epochs, device):
    model.train()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)
    relative_improvement_threshold = 0.01   
    best_loss = float("inf")
    patience = 0
    epochs_no_improvement = 0

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
        scheduler.step(epoch_loss)
        epoch_accuracy = 100 * correct / total
        print(f"epoch: {epoch + 1}, loss: {epoch_loss:.4f}, accuracy: {epoch_accuracy:.2f}%")

        if epoch_loss < best_loss - relative_improvement_threshold:
            best_loss = epoch_loss
            patience = 0
        else:
            patience += 1
        if epochs_no_improvement > patience:
            print("early stopping triggered")
            break


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
    batch_size = 32
    learning_rate = 0.001    
    epochs = 100
    img_size = (28, 28)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    trainloader, testloader = get_dataloaders(batch_size=batch_size, resize=img_size)
    teachercnn_params = {
        "num_filters1": 32,
        "num_filters2": 64,
        "num_filters3": 128,
        "kernel_size1": 1,
        "kernel_size2": 1,
        "kernel_size3": 1,  
        "padding1": 1,
        "padding2": 1,
        "padding3": 1,
        "hidden_units": 128,
        "img_size": img_size
    }
    model = TeacherCNN(**teachercnn_params).to(device) 
    # model = Teacher().to(device)    
    criterion = nn.CrossEntropyLoss()   
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # model.load_state_dict(torch.load("./trained_models/teachercnn_1.pth", map_location=device))
    # model.eval()
    train_teacher(model, trainloader, criterion, optimizer, epochs, device)
    evaluate(model, testloader, device)

    torch.save(model.state_dict(), "./trained_models/teachercnn_1.pth")
    
