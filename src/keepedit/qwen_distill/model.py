from __future__ import annotations

import torch
from torch import nn


class StepDistillAdapter(nn.Module):
    def __init__(self, hidden: int = 48) -> None:
        super().__init__()
        # input, student, student-input residual
        self.net = nn.Sequential(
            nn.Conv2d(9, hidden, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 4, 3, padding=1),
        )

    def forward(self, input_image: torch.Tensor, student_image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = student_image - input_image
        pred = self.net(torch.cat([input_image, student_image, residual], dim=1))
        delta = torch.tanh(pred[:, :3]) * 0.25
        alpha = torch.sigmoid(pred[:, 3:4])
        output = torch.clamp(student_image + alpha * delta, 0.0, 1.0)
        return output, alpha
