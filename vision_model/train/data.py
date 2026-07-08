import os
import gdown
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


URL = "https://drive.google.com/uc?id=1vqX5FKh7bmvxWL0CYjdo2xJFJ4mx3aqd"

CACHE_DIR = os.path.expanduser("~/.cache/cifar10")
FILEPATH = os.path.join(CACHE_DIR, "cifar10.npz")


def cifar_download():
    os.makedirs(CACHE_DIR, exist_ok=True)

    if not os.path.exists(FILEPATH):
        gdown.download(URL, FILEPATH, quiet=False)


MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2470, 0.2435, 0.2616)


train_tf = transforms.Compose([
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.Normalize(MEAN, STD),
])

test_tf = transforms.Compose([
    transforms.Normalize(MEAN, STD),
])


class cifar10(Dataset):
    def __init__(self, split=(80,), part=0, train=True):
        self.train = train
        cifar_download()

        with np.load(FILEPATH) as data:
            images = data["x"]
            labels = data["y"]

        N = len(images)
        n0 = int(split[0] / 100 * N)

        if len(split) == 1:
            self.images = images[:n0]
            self.labels = labels[:n0]
        elif part == 0:
            self.images = images[:n0]
            self.labels = labels[:n0]
        else:
            self.images = images[n0:]
            self.labels = labels[n0:]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.images[idx]).float()

        # CHW, float32, 0-255  ->  CHW, float32, 0-1
        img /= 255.0

        assert img.shape == (3, 32, 32), f"Bad shape: {img.shape}"

        if self.train:
            img = train_tf(img)
        else:
            img = test_tf(img)

        label = torch.as_tensor(self.labels[idx], dtype=torch.long)

        return img, label