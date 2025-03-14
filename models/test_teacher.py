import torch
import torchvision.models as models
from utils import get_dataloaders
from teacher import get_teacher_model


def test_teacher(batch_size=64, device=None):

    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")

    model = get_teacher_model()
    model.to(device)

    _, testloader = get_dataloaders(batch_size=batch_size)

    correct = 0
    total = 0

    with torch.no_grad():
        for img, labels in testloader:
            img, labels = img.to(device), labels.to(device) 
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0) 
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"test accuracy: {accuracy:.2f}%")    
    return accuracy

if __name__ == "__main__":
    test_teacher()



