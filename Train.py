import torch
import time

import numpy as np
import torch.optim as optim

from torch import nn
from matplotlib import pyplot as plt
from SegNet import NeuralNetwork
from VolumeDataset import VolumeDataset
from torch.utils.data import DataLoader


device = "mps" if torch.backends.mps.is_available() else "cpu"
default_device = torch.device("mps")


def init():
    net.zero_grad()


def preview_images(outputs, seg, epoch, i):
    fig = plt.figure(figsize=(6, 6))

    # TODO: Only copy if on MPS
    outputs_local = outputs.cpu()

    output_prev = outputs_local.detach().numpy()[0, 0]
    fig.add_subplot(2, 1, 1)
    plt.imshow((output_prev * 255).astype(np.uint8))
    fig.add_subplot(2, 1, 2)
    plt.imshow(seg[0, 0])
    plt.savefig('output/image_{0}_{1}'.format(epoch, i))
    plt.close()


def train():
    weights = torch.tensor(100).to(default_device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
    optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)

    volume_dataset = VolumeDataset("Training", transform=True)
    test_dataset = VolumeDataset("Validation", transform=True)
    data_loader = DataLoader(volume_dataset, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)

    for epoch in range(10):
        running_loss = 0.0

        for i, data in enumerate(data_loader, 0):
            start = time.time()
            image, seg = data

            optimizer.zero_grad()

            outputs = net(image.to(default_device))

            loss = criterion(outputs, seg.to(default_device))
            # print("Output Total: {0} Seg Total: {1}".format(torch.sum(outputs), torch.sum(seg)))
            # print("Output Max: {0} Seg Max: {1}".format(outputs.max(), seg.max()))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            end = time.time()
            # Monitor
            if i % 10 == 0:
                preview_images(outputs, seg, epoch, i)
                print("Finished step: {0} with loss: {1:2f} - Last Batch Time: {2:2f}s".format(i, loss, end-start))

            # Stats
            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

        print("Starting epoch: {}".format(epoch))

    print("Training complete")


def preview():
    random_in = torch.randn(3, 240, 240).unsqueeze(0)
    out = net(random_in)

    image = out[0].detach().numpy()
    image = image.transpose(1, 2, 0)
    # im = (image * 255).astype(np.uint8)
    plt.imshow(image)
    plt.draw()


if __name__ == "__main__":
    net = NeuralNetwork().to(default_device)
    init()

    train()
