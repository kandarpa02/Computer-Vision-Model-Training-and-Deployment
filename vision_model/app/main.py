from fastapi import FastAPI, UploadFile, File
from .api.predict import model_initiate
from .api.metadata import metadata, class_names
import torch
from PIL import Image
import io
import time

app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model_initiate(device)

@app.get("/")
def home():
    return {"status": "API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    t1 = time.time()
    logits = model(img)
    probs = torch.softmax(logits, dim=1)
    confidence, pred_idx = torch.max(probs, dim=1)
    prediction = class_names[pred_idx.item()]
    confidence = confidence.item()

    t2 = time.time()
    meta = metadata()
    return {
        "prediction": prediction,
        "confidence": confidence,
        **meta,
        "latency_ms": (t2 - t1) * 1000
    } 

@app.post("/predict_batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    images = [
        Image.open(io.BytesIO(await f.read())).convert("RGB")
        for f in files
    ]

    logits = model(images)

    probs = torch.softmax(logits, dim=1)
    confidences, pred_indices = torch.max(probs, dim=1)

    return {
        "predictions": [
            {
                "filename": f.filename,
                "prediction": class_names[idx.item()],
                "confidence": conf.item(),
            }
            for f, idx, conf in zip(files, pred_indices, confidences)
        ]
    }