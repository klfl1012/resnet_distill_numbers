import torch
import torchvision.models as models
import torch.optim as optim, torch.nn as nn


class Teacher(nn.Module):
    def __init__(self):
        super(Teacher, self).__init__()

        pretrained_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model_state_dict = pretrained_model.state_dict()

        self.model = models.resnet18(weights=None)

        for key in list(model_state_dict.keys()):
            if key not in self.model.state_dict() or model_state_dict[key].shape != self.model.state_dict()[key].shape:
                del model_state_dict[key]

        self.model.load_state_dict(model_state_dict, strict=False)

        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(512, 10) 

    def forward(self, x):
        return self.model(x)
    
    def extract_features(self, x, layers=["layer4"]):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        features = {}
        x = self.model.layer1(x)
        if "layer1" in layers:
            features["layer1"] = x

        x = self.model.layer2(x)
        if "layer2" in layers:  
            features["layer2"] = x

        x = self.model.layer3(x)
        if "layer3" in layers:  
            features["layer3"] = x

        x = self.model.layer4(x)
        if "layer4" in layers:  
            features["layer4"] = x

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        features["final"] = x

        return features


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"  
    model = Teacher().to(device)
    model.load_state_dict(torch.load("./trained_models/trained_teacher_state_dict.pth", map_location=device))
    model.eval()
    features = model.exrtact_features(torch.randn(1, 1, 112, 112).to(device))
    print(features.shape)
    print(model)