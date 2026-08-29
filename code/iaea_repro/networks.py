from __future__ import annotations

import torch
from torch import nn


class ResidualFluxNet(nn.Module):
    """Two-module tanh ResNet used by Yang et al. (2023)."""

    def __init__(self, width: int = 20, outputs: int = 1):
        super().__init__()
        self.fc1 = nn.Linear(2, width)
        self.fc2 = nn.Linear(width, width)
        self.skip1 = nn.Linear(2, width)
        self.fc3 = nn.Linear(width, width)
        self.fc4 = nn.Linear(width, width)
        self.skip2 = nn.Linear(width, width)
        self.output = nn.Linear(width, outputs)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        scaled = points / 85.0 - 1.0
        first = torch.tanh(self.fc2(torch.tanh(self.fc1(scaled)))) + torch.tanh(self.skip1(scaled))
        second = torch.tanh(self.fc4(torch.tanh(self.fc3(first)))) + torch.tanh(self.skip2(first))
        return self.output(second)

    def flux(self, points: torch.Tensor, subdomains: torch.Tensor | None = None) -> torch.Tensor:
        raw = self(points)
        if subdomains is not None:
            raw = raw.gather(1, subdomains[:, None]).squeeze(1)
        elif raw.shape[1] == 1:
            raw = raw[:, 0]
        else:
            raise ValueError("Subdomain indices are required for a multi-output network.")
        return raw.square()

