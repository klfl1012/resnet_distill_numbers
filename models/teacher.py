import torch
import torchvision.models as models



def get_teacher_model(model_path, device):
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()

    return model


if __name__ == "__main__":
    model_path = "trained_teacher_1.pth"
    device = "mps" if torch.backends.mps.is_available() else "cpu"  
    model = get_teacher_model(model_path, device)
    print(model)