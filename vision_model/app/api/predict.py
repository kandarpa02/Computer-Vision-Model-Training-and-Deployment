from fastapi import FastAPI, UploadFile, File
from utils import model_initiate
import torch
from PIL import Image
import io

app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model_initiate(device)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    pred = model(img)
    return {"prediction": int(pred.item())}