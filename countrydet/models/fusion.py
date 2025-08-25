"""
-----------------------------
Fusion head (learnable)
-----------------------------
"""

import torch
import torch.nn as nn


class FusionHead(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.65))  # CLIP
        self.beta  = nn.Parameter(torch.tensor(0.35))  # OCR
        self.bias  = nn.Parameter(torch.zeros(num_classes))
        self.eps = 1e-8

        self.t_clip = nn.Parameter(torch.tensor(1.0))
        self.t_ocr  = nn.Parameter(torch.tensor(1.0))

    def forward(self, logits_clip: torch.Tensor, logits_ocr: torch.Tensor) -> torch.Tensor:
        p_clip = (logits_clip / self.t_clip.clamp_min(1e-3)).softmax(dim=-1)
        p_ocr  = (logits_ocr  / self.t_ocr.clamp_min(1e-3)).softmax(dim=-1)
        w_clip = self.alpha.sigmoid()
        w_ocr  = self.beta.sigmoid()
        p = w_clip * p_clip + w_ocr * p_ocr
        return (p + self.eps).log() + self.bias
