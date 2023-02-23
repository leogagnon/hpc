import torch.nn as nn
import torch

class MoE(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_experts: int) -> None:
        super().__init__()

        self.experts = nn.ModuleList(
            [nn.Linear(in_features, out_features) for _ in range(n_experts)]
        )
        self.gating = nn.Sequential(
            nn.Linear(in_features, n_experts), nn.Softmax(dim=-1)
        )

    def forward(self, input):

        gates = self.gating(input).unsqueeze(-1)  # b e 1
        experts_out = torch.stack([expert(input) for expert in self.experts]).transpose(
            0, 1
        )  # b e o

        out = torch.sum(gates * experts_out, dim=1)  # b o

        return out
    

NETWORKS = {
    "cnn": [
        nn.Sequential(nn.Conv2d(1, 10, 3), nn.ReLU(), nn.MaxPool2d(2)),
        nn.Sequential(nn.Conv2d(10, 5, 3), nn.ReLU(), nn.Flatten()),
        nn.Sequential(nn.Linear(5 * 11 * 11, 50), nn.ReLU()),
        nn.Sequential(nn.Linear(50, 30), nn.ReLU()),
        nn.Sequential(nn.Linear(30, 10)),
    ],
    "mlp": [
        nn.Sequential(nn.Flatten(), nn.Linear(784, 512), nn.ReLU()),
        nn.Sequential(nn.Linear(512, 124), nn.ReLU()),
        nn.Sequential(nn.Linear(124, 10)),
    ],
    "moe": [
        nn.Sequential(nn.Conv2d(1, 10, 3), nn.ReLU(), nn.MaxPool2d(2)),
        nn.Sequential(nn.Conv2d(10, 5, 3), nn.ReLU(), nn.Flatten()),
        nn.Sequential(MoE(5 * 11 * 11, 50, 5), nn.ReLU()),
        nn.Sequential(MoE(50, 30, 5), nn.ReLU()),
        nn.Sequential(nn.Linear(30, 10)),
    ],
}
