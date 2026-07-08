import torch

def metadata():
    return {
        "model": "ResNet18",
        "dataset": "CIFAR-10",
        "input_shape": [3, 32, 32],
        "num_classes": 10,
        "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
    }

class_names = [
    "airplane",
    "automobile",
    "bird", 
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]
