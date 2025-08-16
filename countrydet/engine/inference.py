"""
-----------------------------
Inference
-----------------------------
"""

import os
import time
from dataclasses import dataclass
from typing import Any

import torch
from torchvision import transforms
from PIL import Image

from countrydet.engine.cfg import IDX_TO_NAME, NUM_CLASSES
from countrydet.models.feature_extractor import CLIPScorer
from countrydet.models.ocr import OCRCountryScorer
from countrydet.models.fusion import FusionHead

try:
    import easyocr  # PyTorch‑based OCR
    from rapidfuzz import fuzz
except Exception as e:
    easyocr = None
    fuzz = None
    print("[WARN] easyocr or rapidfuzz not found. Install with: pip install easyocr rapidfuzz")

# Optional (speed/quality): set torch.backends flags if on GPU
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


@dataclass
class InferConfig:
    weights: str = "best.pt"
    image_size: int = 336
    use_ocr: bool = True

class InferencePipeline:
    def __init__(self, cfg: InferConfig):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.clip_branch = CLIPScorer(image_size=cfg.image_size).to(self.device)
        self.fusion = FusionHead(NUM_CLASSES).to(self.device)
        ckpt = torch.load(cfg.weights, map_location='cpu') if os.path.exists(cfg.weights) else None
        if ckpt:
            self.clip_branch.load_state_dict(ckpt['clip'], strict=False)
            self.fusion.load_state_dict(ckpt['fusion'], strict=False)
        self.clip_branch.eval(); self.fusion.eval()
        self.use_ocr = cfg.use_ocr and (easyocr is not None)
        if self.use_ocr:
            self.ocr = OCRCountryScorer(languages=["en"])  # add more langs as needed
        # torchvision transforms matching CLIP normalization
        self.tf = transforms.Compose([
            transforms.Resize(cfg.image_size),
            transforms.CenterCrop(cfg.image_size),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                                 std=(0.26862954, 0.26130258, 0.27577711)),
        ])

    @torch.no_grad()
    def predict_one(self, image_path: str, return_timing: bool = True) -> dict[str, Any]:
        img = Image.open(image_path).convert('RGB')
        x = self.tf(img).unsqueeze(0).to(self.device, non_blocking=True)
        torch.cuda.synchronize() if self.device.type=="cuda" else None
        t0 = time.perf_counter()
        with torch.autocast(device_type=self.device.type if self.device.type!="mps" 
                            else 'cpu', enabled=(self.device.type=="cuda")):
            logits_clip = self.clip_branch(x)
        logits_ocr = torch.zeros_like(logits_clip)
        ocr_text = None
        if self.use_ocr:
            logits_ocr = self.ocr.score(image_path).to(self.device).unsqueeze(0)
        logits = self.fusion(logits_clip, logits_ocr)
        probs = logits.softmax(dim=-1).squeeze(0)
        topk = torch.topk(probs, k=min(5, NUM_CLASSES))
        torch.cuda.synchronize() if self.device.type=="cuda" else None
        dt = time.perf_counter() - t0
        return {
            'topk_indices': topk.indices.tolist(),
            'topk_probs': [float(p) for p in topk.values],
            'topk_labels': [IDX_TO_NAME[i] for i in topk.indices.tolist()],
            'latency_s': dt if return_timing else None,
        }
