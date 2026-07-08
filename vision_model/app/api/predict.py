from ...train.resnet18 import ResNet18
from torchvision import transforms
import os
import gdown
import torch
import time

URL = "https://drive.google.com/file/d/1fVEDSBNC-JGH9fFN8Wh6LDb8hmALtkP3/view?usp=drive_link"

MODEL_CACHE = os.path.expanduser("~/.cache/cifar10")
FILEPATH = os.path.join(MODEL_CACHE, "resnet18_cifar10.pt")

def model_download():
    os.makedirs(MODEL_CACHE, exist_ok=True)

    if not os.path.exists(FILEPATH):
        gdown.download(URL, FILEPATH, quiet=False)


MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2470, 0.2435, 0.2616)

normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

def model_initiate(device):
    model_download()
    model = ResNet18().to(device)

    checkpoint = torch.load(FILEPATH, map_location=device)
    model.load_state_dict(checkpoint["param_state"])

    model.eval()

    def predict(img, batched=False):
        with torch.no_grad():
            img = normalize(img)
            if not batched:
                img = img.unsqueeze(0)
            logits = model(img.to(device))
            return logits

    return predict 