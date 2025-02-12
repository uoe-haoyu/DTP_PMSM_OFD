import torch
from torch import nn
import os


class Net(nn.Module):
    def __init__(self, pretrain=None):
        super().__init__()
        self.name = os.path.basename(__file__).split('.')[0]
        self.fc_time = nn.Sequential(
            nn.Linear(6*2048, 64*64),
            nn.ReLU(),
            nn.Linear(64*64, 64*8),
            nn.ReLU(),
            nn.Linear(64*8, 19)
        )


        self.fc_input_freqency = nn.Sequential(
            nn.Linear(6*16, 16*16),
            nn.ReLU(),
            nn.Linear(16*16, 16*4),
            nn.ReLU(),
            nn.Linear(16*4, 19)
        )

        self.alpha = nn.Parameter(torch.full((19,), 0.5))

        self.fc_output = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(19, 19)
        )

    def forward(self, input):
        input_time = input[:,:6,:]
        input_freqency = input[:,6:,:16]


        input_time=input_time.view(input_time.size(0),-1)
        input_freqency=input_freqency.contiguous().view(input_freqency.size(0),-1)

        output_time = self.fc_time(input_time)
        output_freqency = self.fc_input_freqency(input_freqency)

        alpha = torch.sigmoid(self.alpha)  # [19]
        fused = alpha * output_time + (1 - alpha) * output_freqency

        output = self.fc_output(fused)

        return output


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net().to(device)
    x = torch.randn(1, 12, 2048).to(device)
    y = net(x)
    print(y.shape)
