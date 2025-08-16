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
        self.alpha = nn.Parameter(torch.tensor(0.65))  # weight for CLIP branch
        self.beta = nn.Parameter(torch.tensor(0.35))   # weight for OCR branch
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, logits_clip: torch.Tensor, logits_ocr: torch.Tensor) -> torch.Tensor:
        # Normalize per‑branch to comparable scales
        p_clip = logits_clip.softmax(dim=-1)
        p_ocr = logits_ocr.softmax(dim=-1)
        p = self.alpha.sigmoid() * p_clip + self.beta.sigmoid() * p_ocr
        # convert back to logits with log
        eps = 1e-8
        logits = (p + eps).log()
        return logits + self.bias