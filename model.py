import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, channels=64, kernel_size=3):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2, bias=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2, bias=True)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return out+x

class Net(nn.Module):
    def __init__(self, input_channels=3, kernel_size=3, channels=64):
        super(Net, self).__init__()

        self.entry = nn.Conv2d(input_channels, channels, kernel_size=kernel_size, padding=kernel_size//2, bias=True)

        self.resblock1 = ResBlock()
        self.resblock2 = ResBlock()
        self.resblock3 = ResBlock()

        self.out = nn.Conv2d(channels, input_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=True)

        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        input = x
        x = self.relu(self.entry(x))
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)

        x = self.out(x)

        return x + input

if __name__ == '__main__':
    input = torch.randn(1,3,32,32)
    net = Net()
    num_params = 0
    for param in net.parameters():
        num_params += param.nelement()
    print('# of params:', num_params)
    out = net(input)