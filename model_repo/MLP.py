import torch
from torch import nn
import os

class Net(nn.Module):
    def __init__(self, pretrain=None):
        super().__init__()
        self.name = os.path.basename(__file__).split('.')[0]
        #dropout 0.2 when train
        self.fc = nn.Sequential(
            nn.Linear(6*2048, 64*64),
            nn.ReLU(),
            nn.Linear(64*64, 64*8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64*8, 19),
        )

    def forward(self, input):
        input = input[:,:6,:]
        input=input.view(input.size(0),-1)
        output = self.fc(input)
        return output


if __name__ == '__main__':

    net = Net('_')
    x = torch.randn(1, 12, 2048)
    y = net(x)
    print(y.shape)
