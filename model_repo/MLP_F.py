import torch
from torch import nn
import os
from einops import rearrange
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
class Net(nn.Module):
    def __init__(self, pretrain=None):
        super().__init__()
        self.name = os.path.basename(__file__).split('.')[0]



        self.fc_input_freqency = nn.Sequential(
            nn.Linear(6*16, 16*16),
            nn.ReLU(),
            nn.Linear(16*16, 16*4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16*4, 19)
        )



    def forward(self, input):
        input_freqency = input[:,6:,:16]
        input_freqency=input_freqency.contiguous().view(input_freqency.size(0),-1)
        output_freqency = self.fc_input_freqency(input_freqency)

        return output_freqency


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net().to(device)
    x = torch.randn(1, 12, 2048).to(device)
    y = net(x)
    print(y.shape)

