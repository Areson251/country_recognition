"""
---------------------------------
OCR‑free branch: OpenCLIP head
---------------------------------
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from countrydet.engine.cfg import COUNTRY_LIST, NUM_CLASSES

try:
    import open_clip
except Exception as e:
    open_clip = None
    print("[WARN] open_clip not found. Install with: pip install open_clip_torch")


PROMPT_TEMPLATES = [
    "a scanned document from {name}",
    # "an identification card issued by {name}",
    "a passport from {name}",
    # "a driving license from {name}",
    "an official document of {name}",
    "document issued in {name}",
]


def build_prompts(country_list: dict[str, list[str]]) -> tuple[list[str], list[tuple[int,int]]]:
    prompts = []
    owners = []  # (country_idx, alias_idx)
    for ci, (_, aliases) in enumerate(country_list.items()):
        for ai, alias in enumerate(aliases):
            for t in PROMPT_TEMPLATES:
                prompts.append(t.format(name=alias))
                owners.append((ci, ai))
    return prompts, owners


class CLIPScorer(nn.Module):
    def __init__(self, model_name: str = "ViT-B-16", pretrained: str = "laion2b_s34b_b88k", image_size: int = 336):
        super().__init__()
        if open_clip is None:
            raise ImportError("open_clip_torch is required for CLIPScorer")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.image_size = image_size
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()
        
        prompts, owners = build_prompts(COUNTRY_LIST)
        self.register_buffer("text_tokens", self.tokenizer(prompts))
        self.owners = owners  # list of (country_idx, alias_idx)
        self.num_classes = NUM_CLASSES
        
        # temperature scaling (learnable during fusion fine‑tuning)
        self.logit_scale = nn.Parameter(torch.ones(1))

        # Set device for model
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        self.device = device

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return logits over classes via max‑pool over aliases/templates per country."""
        text = self.text_tokens.to(self.device)
        with torch.autocast(device_type=self.device.type):
            img_feats = self.model.encode_image(images)
            img_feats = F.normalize(img_feats, dim=-1)
            txt_feats = self.model.encode_text(text)
            txt_feats = F.normalize(txt_feats, dim=-1)
            sims = img_feats @ txt_feats.T  # [B, num_prompts]
            sims = sims * self.logit_scale.exp()
        # Aggregate to class scores (max over all prompts belonging to same class)
        B, P = sims.shape
        class_scores = torch.full((B, self.num_classes), -1e9, device=self.device)
        # For efficiency, precompute indices per class
        if not hasattr(self, "_class_prompt_idx"):
            idx_by_class = [[] for _ in range(self.num_classes)]
            for pi, (ci, ai) in enumerate(self.owners):
                idx_by_class[ci].append(pi)
            self._class_prompt_idx = [torch.tensor(ix, device=self.device, dtype=torch.long) 
                                                                    for ix in idx_by_class]
        for ci in range(self.num_classes):
            class_scores[:, ci], _ = sims[:, self._class_prompt_idx[ci]].max(dim=1)
        return class_scores
