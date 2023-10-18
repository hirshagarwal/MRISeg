import torch

from torch import nn
import torch.nn.functional as F

device = "mps" if torch.backends.mps.is_available() else "cpu"


class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.MaxEn = nn.MaxPool2d(2, stride=2, return_indices=True)

        conv_kernel_size = 5
        conv_padding = 2
        deconv_kernel_size = 5
        deconv_kernel_padding = 2

        self.conv1 = nn.Conv2d(3, 64, kernel_size=conv_kernel_size, padding=conv_padding)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.5)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=conv_kernel_size, padding=conv_padding)
        self.bn2 = nn.BatchNorm2d(64, momentum=0.5)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=conv_kernel_size, padding=conv_padding)
        self.bn3 = nn.BatchNorm2d(128, momentum=0.5)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=conv_kernel_size, padding=conv_padding)
        self.bn4 = nn.BatchNorm2d(128, momentum=0.5)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=conv_kernel_size, padding=conv_padding)
        self.bn5 = nn.BatchNorm2d(256, momentum=0.5)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=conv_kernel_size, padding=conv_padding)
        self.bn6 = nn.BatchNorm2d(256, momentum=0.5)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=conv_kernel_size, padding=conv_padding)
        self.bn7 = nn.BatchNorm2d(256, momentum=0.5)

        self.conv8 = nn.Conv2d(256, 512, kernel_size=conv_kernel_size, padding=conv_padding)
        self.bn8 = nn.BatchNorm2d(512, momentum=0.5)
        self.conv9 = nn.Conv2d(512, 512, kernel_size=conv_kernel_size, padding=conv_padding)
        self.bn9 = nn.BatchNorm2d(512, momentum=0.5)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=conv_kernel_size, padding=conv_padding)
        self.bn10 = nn.BatchNorm2d(512, momentum=0.5)

        self.conv11 = nn.Conv2d(512, 512, kernel_size=conv_kernel_size, padding=conv_padding)
        self.bn11 = nn.BatchNorm2d(512, momentum=0.5)
        self.conv12 = nn.Conv2d(512, 512, kernel_size=conv_kernel_size, padding=conv_padding)
        self.bn12 = nn.BatchNorm2d(512, momentum=0.5)
        self.conv13 = nn.Conv2d(512, 512, kernel_size=conv_kernel_size, padding=conv_padding)
        self.bn13 = nn.BatchNorm2d(512, momentum=0.5)

        # Deconvolution Layers

        # General Max Pool 2D/Upsampling for DECODING layers
        self.MaxDe = nn.MaxUnpool2d(2, stride=2, padding=0)

        self.deconv1 = nn.Conv2d(512, 512, kernel_size=deconv_kernel_size, padding=deconv_kernel_padding)
        self.debn1 = nn.BatchNorm2d(512, momentum=0.5)
        self.deconv2 = nn.Conv2d(512, 512, kernel_size=deconv_kernel_size, padding=deconv_kernel_padding)
        self.debn2 = nn.BatchNorm2d(512, momentum=0.5)
        self.deconv3 = nn.Conv2d(512, 512, kernel_size=deconv_kernel_size, padding=deconv_kernel_padding)
        self.debn3 = nn.BatchNorm2d(512, momentum=0.5)

        self.deconv4 = nn.Conv2d(512, 512, kernel_size=deconv_kernel_size, padding=deconv_kernel_padding)
        self.debn4 = nn.BatchNorm2d(512, momentum=0.5)
        self.deconv5 = nn.Conv2d(512, 512, kernel_size=deconv_kernel_size, padding=deconv_kernel_padding)
        self.debn5 = nn.BatchNorm2d(512, momentum=0.5)
        self.deconv6 = nn.Conv2d(512, 256, kernel_size=deconv_kernel_size, padding=deconv_kernel_padding)
        self.debn6 = nn.BatchNorm2d(256, momentum=0.5)

        self.deconv7 = nn.Conv2d(256, 256, kernel_size=deconv_kernel_size, padding=deconv_kernel_padding)
        self.debn7 = nn.BatchNorm2d(256, momentum=0.5)
        self.deconv8 = nn.Conv2d(256, 256, kernel_size=deconv_kernel_size, padding=deconv_kernel_padding)
        self.debn8 = nn.BatchNorm2d(256, momentum=0.5)
        self.deconv9 = nn.Conv2d(256, 128, kernel_size=deconv_kernel_size, padding=deconv_kernel_padding)
        self.debn9 = nn.BatchNorm2d(128, momentum=0.5)

        self.deconv10 = nn.Conv2d(128, 128, kernel_size=deconv_kernel_size, padding=deconv_kernel_padding)
        self.debn10 = nn.BatchNorm2d(128, momentum=0.5)
        self.deconv11 = nn.Conv2d(128, 64, kernel_size=deconv_kernel_size, padding=deconv_kernel_padding)
        self.debn11 = nn.BatchNorm2d(64, momentum=0.5)

        self.deconv12 = nn.Conv2d(64, 64, kernel_size=deconv_kernel_size, padding=deconv_kernel_padding)
        self.debn12 = nn.BatchNorm2d(64, momentum=0.5)
        self.deconv13 = nn.Conv2d(64, 1, kernel_size=deconv_kernel_size, padding=deconv_kernel_padding)
        self.debn13 = nn.BatchNorm2d(1, momentum=0.5)

        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x, ind1 = self.MaxEn(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x, ind2 = self.MaxEn(x)

        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x, ind3 = self.MaxEn(x)

        x = F.relu(self.bn8(self.conv8(x)))
        x = F.relu(self.bn9(self.conv9(x)))
        x = F.relu(self.bn10(self.conv10(x)))
        x, ind4 = self.MaxEn(x)
        size4 = x.size()

        x = F.relu(self.bn11(self.conv11(x)))
        x = F.relu(self.bn12(self.conv12(x)))
        x = F.relu(self.bn13(self.conv13(x)))
        x, ind5 = self.MaxEn(x)

        # Decode
        x = self.MaxDe(x, ind5, output_size=size4)
        x = F.relu(self.debn1(self.deconv1(x)))
        x = F.relu(self.debn2(self.deconv2(x)))
        x = F.relu(self.debn3(self.deconv3(x)))

        x = self.MaxDe(x, ind4)
        x = F.relu(self.debn4(self.deconv4(x)))
        x = F.relu(self.debn5(self.deconv5(x)))
        x = F.relu(self.debn6(self.deconv6(x)))

        x = self.MaxDe(x, ind3)
        x = F.relu(self.debn7(self.deconv7(x)))
        x = F.relu(self.debn8(self.deconv8(x)))
        x = F.relu(self.debn9(self.deconv9(x)))

        x = self.MaxDe(x, ind2)
        x = F.relu(self.debn10(self.deconv10(x)))
        x = F.relu(self.debn11(self.deconv11(x)))

        x = self.MaxDe(x, ind1)
        x = F.relu(self.debn12(self.deconv12(x)))
        x = F.relu(self.debn13(self.deconv13(x)))
        # x = F.softmax(x, dim=2)
        x = self.sig(x)
        # x = torch.as_tensor((x > 0.5), dtype=torch.float32)
        # print(x)
        return x
