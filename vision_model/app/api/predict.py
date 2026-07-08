from ...train.resnet18 import ResNet18
from torchvision import transforms
import os
import gdown
import torch
import time
import zipfile

URL = "https://drive.google.com/file/d/1bHZGVngZpKAQ5TFc0vsz8y5Xm2RQPuJ-/view?usp=sharing"

MODEL_CACHE = os.path.expanduser("~/.cache/cifar10")

ZIP_PATH = os.path.join(MODEL_CACHE, "ckpt183.zip")
PT_PATH = os.path.join(MODEL_CACHE, "ckpt183.pt")

def model_download():
    os.makedirs(MODEL_CACHE, exist_ok=True)

    # Download only if the extracted checkpoint doesn't already exist
    if not os.path.exists(PT_PATH):
        gdown.download(URL, ZIP_PATH, quiet=False)

        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(MODEL_CACHE)

        os.remove(ZIP_PATH)

    return PT_PATH

MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2470, 0.2435, 0.2616)

normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

def model_initiate(device):
    model_download()
    model = ResNet18().to(device)

    checkpoint = torch.load(PT_PATH, map_location=device)
    model.load_state_dict(checkpoint["param_state"])
    model.eval()

    def predict(images):
        with torch.no_grad():

            # Single image
            if not isinstance(images, list):
                images = [images]

            batch = torch.stack([
                normalize(img) for img in images
            ]).to(device)

            logits = model(batch)
            return logits

    return predict