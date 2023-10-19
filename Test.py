from pathlib import Path

import torch
from torch.utils.data import DataLoader

from VolumeDataset import VolumeDataset
from SegNet import NeuralNetwork

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
default_device = torch.device(device)


def init():
    net.zero_grad()

    model_path = Path("model/model.pth")
    if model_path.is_file():
        checkpoint = torch.load("model/model.pth")
        net.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded")


def compute_accuracy(output, seg):
    computed = torch.logical_and(output, seg)
    print(computed.shape)

def test():
    net.eval()
    test_dataset = VolumeDataset("Validation", transform=True)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)

    running_accuracy = 0
    for index, data in enumerate(test_loader):
        net.eval()
        image, seg = data

        outputs = net(image.to(default_device))
        accuracy = compute_accuracy(outputs, seg)


if __name__ == "__main__":
    net = NeuralNetwork().to(default_device)
    init()
    test()
