import os
import torch
import torch.nn as nn

import os
import torch
import torch.nn as nn

class Net(nn.Module):
    """Fault-diagnosis network with gated attention-based feature fusion."""
    def __init__(self, pretrain=None):
        super().__init__()
        self.name = os.path.basename(__file__).split('.')[0]

        # ───────────── Time-domain branch ─────────────
        self.fc_time = nn.Sequential(
            nn.Linear(6 * 2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, 19)
        )

        # ───────────── Frequency-domain branch ─────────────
        self.fc_freq = nn.Sequential(
            nn.Linear(6 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 19)
        )

        # ───────────── Gated Attention Fusion ─────────────
        self.gate = nn.Sequential(
            nn.Linear(19 * 2, 30),
            nn.ReLU(),
            nn.Linear(30, 19),
            nn.Sigmoid()  # Output gate ∈ [0, 1]
        )

        # ───────────── Classification head ─────────────
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(19, 19)
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, 12, L)
        Returns:
            logits: Tensor of shape (batch, 19)
        """
        # Split modalities
        x_time = x[:, :6, :].contiguous().view(x.size(0), -1)
        x_freq = x[:, 6:, :16].contiguous().view(x.size(0), -1)

        # Two branches
        out_time = self.fc_time(x_time)   # [B, 19]
        out_freq = self.fc_freq(x_freq)   # [B, 19]

        # Gated attention
        gate_input = torch.cat([out_time, out_freq], dim=1)  # [B, 38]
        g = self.gate(gate_input)                            # [B, 19]
        fused = g * out_time + (1 - g) * out_freq            # [B, 19]

        # Final classification
        logits = self.head(fused)
        return logits

if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis, parameter_count,parameter_count_table


    # 将 FLOPs 转换为 K（千）、M（百万）、G（十亿）等单位
    def format_flops(flops):
        if flops >= 1e3:
            return f"{flops / 1e6:.2f} MFLOPs"  # 转换为百万（Mega）
        else:
            return f"{flops} FLOPs"


    # 将参数量转换为 K（千）、M（百万）等单位
    def format_params(params):
        if params >= 1e3:
            return f"{params / 1e6:.3f} MParams"  # 转换为百万（Mega）
        else:
            return f"{params} Params"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net().to(device)

    # 创建输入数据
    x = torch.randn(1, 12, 2048).to(device)
    y = net(x)
    print(y.shape)

    # 计算参数量
    param_count = parameter_count(net)
    total_params = sum(param_count[k] for k in param_count)
    formatted_params = format_params(total_params)
    print(f"Total Parameters: {formatted_params}")

    # 计算 FLOPs
    flop_analyzer = FlopCountAnalysis(net, x)
    total_flops = flop_analyzer.total()
    formatted_flops = format_flops(total_flops)
    print(f"Total FLOPs: {formatted_flops}")