from ..train.resnet18 import ResNet18
from ..train.data import test_tf
import os
import gdown
import torch

URL = "https://drive.google.com/file/d/1fVEDSBNC-JGH9fFN8Wh6LDb8hmALtkP3/view?usp=drive_link"
PATH = "vision_model/model_inference/resnet18_cifar10.pt"

if not os.path.exists(PATH):
    gdown.download(URL, output=PATH, quiet=False)

def model_initiate(device):
    model = ResNet18().to(device)

    checkpoint = torch.load(PATH, map_location=device)
    model.load_state_dict(checkpoint["param_state"])

    model.eval()

    def predict(img, batched=False):
        with torch.no_grad():
            img = test_tf(img)
            if not batched:
                img = img.unsqueeze(0)
            logits = model(img.to(device))
            return torch.argmax(logits, dim=-1)

    return predict 