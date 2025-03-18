import onnxruntime as ort
import json
from PIL import Image
from torchvision import transforms
import torch

class ModelManager:

    def __init__(self, config_paths="./models/configs/models_config.json"):
        self.model_paths = self.load_config(config_paths)["models"]
        self.models = {}
        self.providers, self.device = self.get_available_device()

    def load_config(self, config_path):
        with open(config_path, "r") as f:
            return json.load(f)

    def get_available_device(self):
        available_providers = ort.get_available_providers()

        if "MPSExecutionProvider" in available_providers:
            return ["MPSExecutionProvider"], "mps"
        elif "CUDAExecutionProvider" in available_providers:
            return ["CUDAExecutionProvider"], "cuda"
        else:
            return ["CPUExecutionProvider"], "cpu"

    def load_model(self, model_name):
        if model_name not in self.models:
            if model_name in self.model_paths:
                model_path = self.model_paths[model_name]
                self.models[model_name] = ort.InferenceSession(model_path, providers=self.providers)
            else:
                raise ValueError(f"Model {model_name} not found in config file")
            
        return self.models[model_name]
    
    def transform_img(self, img, resize=(28, 28)):
        img = Image.fromarray(img)
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1), 
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])

        return transform(img).unsqueeze(0).to(self.device)

    def predict(self, model_name, img_tensor):
        if model_name not in self.models:
            self.load_model(model_name)
        
        model = self.models[model_name]
        img_numpy = img_tensor.cpu().numpy().astype("float32")  

        out = model.run(None, {"input": img_numpy})[0]

        probs = torch.nn.functional.softmax(torch.tensor(out), dim=1).cpu().numpy()
        return probs
        