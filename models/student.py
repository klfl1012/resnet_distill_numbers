import torch
import torch.nn as nn 

class StudentCNN(nn.Module):

    def __init__(self, num_filters1, num_filters2, num_filters3, kernel_size1, kernel_size2, kernel_size3, padding1, padding2, padding3, hidden_units, img_size=(28, 28)):
        super(StudentCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, num_filters1, kernel_size=kernel_size1, padding=padding1)
        self.conv2 = nn.Conv2d(num_filters1, num_filters2, kernel_size=kernel_size2, padding=padding2)
        self.conv3 = nn.Conv2d(num_filters2, num_filters3, kernel_size=kernel_size3, padding=padding3)
        self.pool = nn.MaxPool2d(2, 2)

        self._to_linear = self._compute_flattened_size(img_size[0], img_size[1])

        self.fc1 = nn.Linear(self._to_linear, hidden_units)
        self.fc2 = nn.Linear(hidden_units, 10)

    def _compute_flattened_size(self, h, w):
        with torch.no_grad():
            x = torch.randn(1, 1, h, w)
            x = self.pool(nn.functional.relu(self.conv1(x)))
            x = self.pool(nn.functional.relu(self.conv2(x)))
            x = self.pool(nn.functional.relu(self.conv3(x)))
            x = x.view(x.size(0), -1)

            return x.size(1)

    def extract_features(self, x, layers=["final"]):
        features = {}   

        x = self.pool(nn.functional.relu(self.conv1(x)))
        if "conv1" in layers:
            features["conv1"] = x
        
        x = self.pool(nn.functional.relu(self.conv2(x)))
        if "conv2" in layers:   
            features["conv2"] = x

        x = self.pool(nn.functional.relu(self.conv3(x)))
        if "conv3" in layers:
            features["conv3"] = x

        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        features["final"] = x
        
        return features 

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)    


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
    pass    
    