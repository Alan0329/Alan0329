import torch
import torch.nn as nn

# https://www.youtube.com/watch?v=2-zGCx4iv_k 8:30


class Dueling_Q_Network(nn.Module):
    def __init__(self, input_size):
        super(Dueling_Q_Network, self).__init__()
        self.input_size = input_size
        print("input", self.input_size)
        self.backbone = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=4,
                            kernel_size=2, dilation=1, bias=True, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(4),
            # torch.nn.Conv1d(in_channels=2, out_channels=4,
            #                 kernel_size=2, dilation=2, bias=True),
            # torch.nn.ReLU(),
            # torch.nn.BatchNorm1d(4),
            # torch.nn.Dropout(0.3),
            torch.nn.Conv1d(in_channels=4, out_channels=8,
                            kernel_size=2, dilation=4, bias=True),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(8),
            torch.nn.Dropout(0.3),
            torch.nn.Conv1d(in_channels=8, out_channels=16,
                            kernel_size=2, dilation=8, bias=True),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(16),
            # torch.nn.Dropout(0.3),
            torch.nn.Conv1d(in_channels=16, out_channels=64,
                            kernel_size=2, dilation=16, bias=True, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(64),
            # torch.nn.Conv1d(in_channels=64, out_channels=128,
            #                 kernel_size=3, dilation=32, bias=True, padding=1),
            # torch.nn.ReLU(),
            # torch.nn.BatchNorm1d(128),
            # torch.nn.Conv1d(in_channels=64, out_channels=128,
            #                 kernel_size=2, dilation=64, bias=True),
            # torch.nn.ReLU(),
            # torch.nn.BatchNorm1d(128)
        )

        self.state_value = torch.nn.Sequential(
            torch.nn.Linear(1920, 1024),
            torch.nn.SELU(),
            # torch.nn.Linear(2048, 1024),
            # torch.nn.SELU(),
            # torch.nn.AlphaDropout(0.3),
            torch.nn.Linear(1024, 128),
            torch.nn.SELU(),
            torch.nn.Linear(128, 1),
            torch.nn.SELU()
        )
        self.state_value = torch.nn.Sequential(
            torch.nn.Linear(1920, 1024),
            torch.nn.SELU(),
            # torch.nn.Linear(2048, 1024),
            # torch.nn.SELU(),
            # torch.nn.AlphaDropout(0.3),
            torch.nn.Linear(1024, 128),
            torch.nn.SELU(),
            torch.nn.Linear(128, 1),  # ??????state??????value
            torch.nn.SELU()
        )

        self.advantage_value = torch.nn.Sequential(
            torch.nn.Linear(1920, 1024),
            torch.nn.SELU(),
            # torch.nn.Linear(2048, 1024),
            # torch.nn.SELU(),
            # torch.nn.AlphaDropout(0.3),
            torch.nn.Linear(1024, 128),
            torch.nn.SELU(),
            torch.nn.Linear(128, 3),  # ??????state,action??????value ??????????????????3???value
            torch.nn.SELU()
        )

    def reset(self):
        self.zero_grad()

    def forward(self, x):
        x = self.backbone(x.view(-1, 1, self.input_size))
        state_value = self.state_value(x.view(-1, 1920))
        advantage_value = self.advantage_value(x.view(-1, 1920))
        advantage_mean = torch.Tensor.mean(
            advantage_value, dim=1, keepdim=True)  # keepdim ???????????????????????????????????????????????????????????????

        q_value = state_value.expand(
            [-1, 3]) + (advantage_value - advantage_mean.expand([-1, 3]))  # v(s)+A(s,a)

        return q_value
