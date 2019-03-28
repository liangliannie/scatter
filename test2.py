import torch
import torch.nn as nn
import torch.functional as F

class Net(nn.Module):

    def __init__(self, opts):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        input_channel = 1
        output_channel = 1
        kernel_size = 3

        self.conv1 = self.conv_bn_relu(64*520, 520)
        self.conv2 = self.conv_bn_relu(520, 256)
        self.conv3 = self.conv_bn_relu(256, 128)
        self.conv4 = self.conv_bn_relu(128, 256)
        self.conv5 = self.conv_bn_relu(256, 520)
        self.conv6 = self.conv_bn_relu(520, 64*520)
        # self.conv7 = [nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=True)]


        # an affine operation: y = Wx + b
        # alllayers = self.conv1 + self.conv6
        alllayers = self.conv1 + self.conv2+self.conv3+self.conv4+ self.conv5+self.conv6

        self.layer = torch.nn.Sequential(*alllayers)

    def conv_bn_relu(self, num_features_in, num_features_out):
        layers = []
        layers.append(nn.Conv2d(num_features_in, num_features_out, kernel_size=3, stride=1, padding=1, bias=True))
        layers.append(nn.BatchNorm2d(num_features_out))
        layers.append(nn.ReLU())
        return layers

    def lin_tan_drop(self, num_features_in, num_features_out, dropout=0.5):
        layers = []
        layers.append(nn.Linear(num_features_in, num_features_out, bias=True))
        layers.append(nn.Tanh())
        layers.append(nn.Dropout(p=dropout))
        return layers

    def forward(self, x):
        x = x.reshape(-1, 64*520)
        x = self.layer(x)
        x = x.reshape(-1, 1, 64, 520)
        x = torch.sigmoid(x)
        return x







if __name__ == '__main__':
    opts = 'Hello'
    network = Net(opts)
    x = torch.randn(1,1,50,520)
    out = network.forward(x)
    print(out.shape)

