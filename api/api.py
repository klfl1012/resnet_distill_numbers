from fastapi import FastAPI, HTTPException    
import onnxruntime as ort, numpy as np, cv2, torch, json
from loadmodels import ModelManager
from pydantic import BaseModel

app = FastAPI()
manager = ModelManager(config_paths="./models/configs/models_config.json")

@app.get("/")
def read_root():
    return {"message": "Model Inference API is running"}   

@app.get("/models/")
def list_models():
    return {"available models": list(manager.model_paths.keys())}

class ImageRequest(BaseModel):
    image: list
    model_list: list = ["teacher", "student"]


@app.post("/predict/")
async def predict(request: ImageRequest):
    try:
        img_numpy = np.array(request.image, dtype=np.float32)
        img_tensor = torch.tensor(img_numpy).unsqueeze(0)   

        predictions = {}
        for model_name in request.model_list:
            if model_name in manager.model_paths:
                probs = manager.predict(model_name, img_tensor).tolist()
                predictions[model_name] = probs[0]

            else:
                raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

        return {"predictions": predictions}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))