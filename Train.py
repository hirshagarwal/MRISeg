from pathlib import Path

import torch
import time

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

    model_path = Path("model/model.pth")
    if model_path.is_file():
        checkpoint = torch.load("model/model.pth")
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Model loaded")


def save():
    torch.save({
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, "model/model.pth")
    print("Model Saved")


def preview_images(image, outputs, seg, epoch, i):
    fig = plt.figure(figsize=(9, 9))
    fig.set_figheight(12)
    fig.set_figwidth(12)
    # TODO: Only copy if on MPS
    outputs_local = outputs.cpu()
    image_local = image.cpu()
    output_count = 5
    for idx in range(1, output_count + 1):
        input_prev = image_local.detach().numpy()[idx]

        ax = fig.add_subplot(5, output_count, idx)
        ax.title.set_text("Input {} (T1)".format(idx))
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        plt.imshow(input_prev[0])

        ax1 = fig.add_subplot(5, output_count, idx + output_count)
        ax1.title.set_text("Input {} (T2)".format(idx))
        ax1.set_yticklabels([])
        ax1.set_xticklabels([])
        plt.imshow(input_prev[1])

        ax2 = fig.add_subplot(5, output_count, idx + (2 * output_count))
        ax2.title.set_text("Input {} (FLAIR)".format(idx))
        ax2.set_yticklabels([])
        ax2.set_xticklabels([])
        plt.imshow(input_prev[2])

        ax3 = fig.add_subplot(5, output_count, idx + (3 * output_count))
        ax3.title.set_text("Seg {}".format(idx))
        ax3.set_yticklabels([])
        ax3.set_xticklabels([])
        plt.imshow(seg[idx, 0])

        output_prev = outputs_local.detach().numpy()[idx, 0]
        ax4 = fig.add_subplot(5, output_count, idx + (4 * output_count))
        ax4.title.set_text("Output {}".format(idx))
        ax4.set_yticklabels([])
        ax4.set_xticklabels([])
        plt.imshow(output_prev)

    plt.savefig('output/image_{0}_{1}'.format(epoch, i))
    plt.close()


def train():
    pos_weight = torch.tensor(25).to(default_device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

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
            if i % 100 == 0:
                preview_images(image, outputs, seg, epoch, i)
                save()
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
    plt.imshow(image)
    plt.draw()


if __name__ == "__main__":
    net = NeuralNetwork().to(default_device)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    init()

    train()
