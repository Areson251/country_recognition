"""
Country-from-Document (PyTorch) — End‑to‑End Pipeline
=====================================================
A practical, fast (<1s/image on modern GPU), and extensible pipeline to infer a document's country
from scanned images of passports/IDs/drivers licenses/etc.

Key ideas
---------
1) **OCR‑free branch (vision-language)**: use a CLIP‑like image encoder + text prompts with country names
   to score image–text similarity. This generalizes across layouts and works even without MRZ.
2) **OCR branch (keyword+MRZ)**: lightweight OCR (EasyOCR) to pull text; regex + fuzzy match for
   country names and MRZ issuing state codes (ISO‑3166‑1 alpha‑3 style).
3) **Score fusion**: a tiny learnable layer that fuses both branches' scores.

You can:
- Train (fine‑tune temperature/weights) on your 24‑class dataset
- Evaluate with top‑1/top‑3 accuracy and confusion matrix
- Run optimized inference (<1s/img with AMP, NHWC, judicious image sizes)

Notes
-----
- Replace `COUNTRY_LIST`/`ALPHA3_MAP` with your exact 24 countries and aliases.
- Works with **hundreds** of countries by just adding prompt strings; retraining optional.
- EasyOCR is PyTorch‑based; install: `pip install easyocr open_clip_torch rapidfuzz`.

"""
from __future__ import annotations
import os
import re
import time
import glob
import math
import json
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Optional (speed/quality): set torch.backends flags if on GPU
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# ---------------------------
# 1) Config & Country metadata
# ---------------------------

# Example country list with aliases; fill with your 24 countries
COUNTRY_LIST: Dict[str, List[str]] = {
    "Germany": ["Germany", "Deutschland", "Bundesrepublik Deutschland", "DEU", "GER"],
    "France": ["France", "République française", "FRA"],
    "Spain": ["Spain", "España", "Reino de España", "ESP"],
    "Italy": ["Italy", "Italia", "Repubblica Italiana", "ITA"],
    "United Kingdom": ["United Kingdom", "United Kingdom of Great Britain and Northern Ireland", "UK", "GBR", "Great Britain"],
    "United States": ["United States", "United States of America", "USA", "United States of America"],
    "Canada": ["Canada", "CAN"],
    "Australia": ["Australia", "AUS"],
    "Netherlands": ["Netherlands", "Nederland", "NLD"],
    "Poland": ["Poland", "Polska", "POL"],
    "Portugal": ["Portugal", "PRT"],
    "Greece": ["Greece", "Ελλάδα", "GRC"],
    "Sweden": ["Sweden", "Sverige", "SWE"],
    "Norway": ["Norway", "Norge", "NOR"],
    "Denmark": ["Denmark", "Danmark", "DNK"],
    "Finland": ["Finland", "Suomi", "FIN"],
    "Ireland": ["Ireland", "Éire", "IRL"],
    "Switzerland": ["Switzerland", "Schweiz", "Suisse", "Svizzera", "CHE"],
    "Austria": ["Austria", "Österreich", "AUT"],
    "Belgium": ["Belgium", "Belgique", "België", "BEL"],
    "Czechia": ["Czechia", "Czech Republic", "Česko", "CZE"],
    "Hungary": ["Hungary", "Magyarország", "HUN"],
    "Romania": ["Romania", "România", "ROU"],
    "Bulgaria": ["Bulgaria", "България", "BGR"],
}

# MRZ issuing state (ISO 3166-1 alpha-3) map for the 24 countries
ALPHA3_MAP = {
    "DEU": "Germany", "FRA": "France", "ESP": "Spain", "ITA": "Italy", "GBR": "United Kingdom",
    "USA": "United States", "CAN": "Canada", "AUS": "Australia", "NLD": "Netherlands",
    "POL": "Poland", "PRT": "Portugal", "GRC": "Greece", "SWE": "Sweden", "NOR": "Norway",
    "DNK": "Denmark", "FIN": "Finland", "IRL": "Ireland", "CHE": "Switzerland", "AUT": "Austria",
    "BEL": "Belgium", "CZE": "Czechia", "HUN": "Hungary", "ROU": "Romania", "BGR": "Bulgaria",
}

COUNTRIES = list(COUNTRY_LIST.keys())
NUM_CLASSES = len(COUNTRIES)
NAME_TO_IDX = {n: i for i, n in enumerate(COUNTRIES)}
IDX_TO_NAME = {i: n for n, i in NAME_TO_IDX.items()}

# ---------------------------
# 2) Dataset & Augmentations
# ---------------------------

class DocDataset(Dataset):
    """Assumes a folder structure: root/COUNTRY_NAME/*.jpg (or png)
    Example: root/France/0001.jpg
    """
    def __init__(self, root: str, split: str = "train", image_size: int = 336, val_ratio: float = 0.15, seed: int = 42):
        self.root = root
        self.image_size = image_size
        self.split = split
        random.seed(seed)
        self.samples: List[Tuple[str, int]] = []
        for cname in COUNTRIES:
            files = []
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
                files.extend(glob.glob(os.path.join(root, cname, ext)))
            files.sort()
            if not files:
                print("No files detected")
                continue
            # deterministic split per class
            idx = int(len(files) * (1 - val_ratio))
            if split == "train":
                chosen = files[:idx]
            else:
                chosen = files[idx:]
            self.samples += [(f, NAME_TO_IDX[cname]) for f in chosen]
        
        # Augmentations
        if split == "train":
            self.tf = transforms.Compose([
                transforms.Resize(int(image_size * 1.1)),
                transforms.RandomRotation( ( -4, 4 ) ),
                transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
                transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.05),
                transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
                transforms.RandomAdjustSharpness(1.5, p=0.2),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),  # CLIP norm
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fp, y = self.samples[idx]
        img = Image.open(fp).convert("RGB")
        x = self.tf(img)
        return x, y, fp

# ---------------------------------
# 3) OCR‑free branch: OpenCLIP head
# ---------------------------------

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

def build_prompts(country_list: Dict[str, List[str]]) -> Tuple[List[str], List[Tuple[int,int]]]:
    prompts = []
    owners = []  # (country_idx, alias_idx)
    for ci, (cname, aliases) in enumerate(country_list.items()):
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

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return logits over classes via max‑pool over aliases/templates per country."""
        device = images.device
        text = self.text_tokens.to(device)
        with torch.autocast(device_type=device.type if device.type != 'mps' else 'cpu', enabled=(device.type=="cuda")):
            img_feats = self.model.encode_image(images)
            img_feats = F.normalize(img_feats, dim=-1)
            txt_feats = self.model.encode_text(text)
            txt_feats = F.normalize(txt_feats, dim=-1)
            sims = img_feats @ txt_feats.T  # [B, num_prompts]
            sims = sims * self.logit_scale.exp()
        # Aggregate to class scores (max over all prompts belonging to same class)
        B, P = sims.shape
        class_scores = torch.full((B, self.num_classes), -1e9, device=device)
        # For efficiency, precompute indices per class
        if not hasattr(self, "_class_prompt_idx"):
            idx_by_class = [[] for _ in range(self.num_classes)]
            for pi, (ci, ai) in enumerate(self.owners):
                idx_by_class[ci].append(pi)
            self._class_prompt_idx = [torch.tensor(ix, device=device, dtype=torch.long) for ix in idx_by_class]
        for ci in range(self.num_classes):
            class_scores[:, ci], _ = sims[:, self._class_prompt_idx[ci]].max(dim=1)
        return class_scores

# -----------------------------
# 4) OCR branch (EasyOCR + heuristics)
# -----------------------------

try:
    import easyocr  # PyTorch‑based OCR
    from rapidfuzz import fuzz
except Exception as e:
    easyocr = None
    fuzz = None
    print("[WARN] easyocr or rapidfuzz not found. Install with: pip install easyocr rapidfuzz")

# MRZ_ISSUING_RE = re.compile(r"[IPV]\<([A-Z]{3})")  # matches e.g., P<DEU

class OCRCountryScorer:
    def __init__(self, languages: List[str] = ["en"]):
        if easyocr is None:
            raise ImportError("easyocr is required for OCRCountryScorer")
        self.reader = easyocr.Reader(languages, gpu=torch.cuda.is_available())
        # Flatten alias dict to (country_idx, alias_text)
        self.alias_list: List[Tuple[int, str]] = []
        for cname, aliases in COUNTRY_LIST.items():
            ci = NAME_TO_IDX[cname]
            for a in aliases:
                self.alias_list.append((ci, a.lower()))

    @torch.no_grad()
    def score(self, image_path: str) -> torch.Tensor:
        """Return logits over classes based on OCR keyword + MRZ cues."""
        text_items = self.reader.readtext(image_path, detail=0, paragraph=True)
        text_full = " \n ".join(text_items)
        text_lower = text_full.lower()
        scores = torch.zeros(NUM_CLASSES)
        # # 1) MRZ issuing state code
        # m = MRZ_ISSUING_RE.search(text_full.replace(" ", ""))
        # if m:
        #     code = m.group(1)
        #     country = ALPHA3_MAP.get(code)
        #     if country is not None:
        #         scores[NAME_TO_IDX[country]] += 2.0  # strong signal
        # 2) Alias fuzzy matches
        if fuzz is not None:
            for ci, alias in self.alias_list:
                s = fuzz.partial_ratio(alias, text_lower) / 100.0
                if s > 0.6:
                    scores[ci] += s
        else:
            # exact contains fallback
            for ci, alias in self.alias_list:
                if alias in text_lower:
                    scores[ci] += 0.7
        return scores

# -----------------------------
# 5) Fusion head (learnable)
# -----------------------------

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

# -----------------------------
# 6) Training (fine‑tune fusion)
# -----------------------------

@dataclass
class TrainConfig:
    root: str
    image_size: int = 336
    batch_size: int = 16
    epochs: int = 10
    lr: float = 1e-3
    num_workers: int = 4
    mixed_precision: bool = True

class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_ds = DocDataset(cfg.root, split="train", image_size=cfg.image_size)
        self.val_ds = DocDataset(cfg.root, split="val", image_size=cfg.image_size)
        self.train_dl = DataLoader(self.train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
        self.val_dl = DataLoader(self.val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
        
        self.clip_branch = CLIPScorer(image_size=cfg.image_size).to(self.device)
        self.fusion = FusionHead(NUM_CLASSES).to(self.device)
        self.opt = torch.optim.AdamW(list(self.fusion.parameters()) + [self.clip_branch.logit_scale], lr=cfg.lr, weight_decay=1e-4)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type=="cuda" and cfg.mixed_precision))

    def _step(self, batch, train: bool):
        x, y, _ = batch
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        with torch.autocast(device_type=self.device.type if self.device.type!="mps" else 'cpu', enabled=(self.device.type=="cuda" and self.cfg.mixed_precision)):
            logits_clip = self.clip_branch(x)
            # For training, approximate OCR scores as zeros (since OCR uses file paths).
            # We keep fusion learnable mainly over CLIP + bias; OCR is added at inference.
            logits = self.fusion(logits_clip, torch.zeros_like(logits_clip))
            loss = F.cross_entropy(logits, y)
        if train:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad(set_to_none=True)
        return loss.detach().item(), logits.detach()

    @torch.no_grad()
    def _eval_epoch(self):
        self.clip_branch.eval(); self.fusion.eval()
        n, correct, correct_top3 = 0, 0, 0
        all_preds, all_tgts = [], []
        for x, y, _ in self.val_dl:
            x = x.to(self.device)
            y = y.to(self.device)
            logits = self.fusion(self.clip_branch(x), torch.zeros((x.size(0), NUM_CLASSES), device=self.device))
            top1 = logits.argmax(dim=-1)
            top3 = logits.topk(3, dim=-1).indices
            correct += (top1==y).sum().item()
            correct_top3 += (top3==y.unsqueeze(1)).any(dim=1).sum().item()
            n += x.size(0)
            all_preds.append(top1.cpu())
            all_tgts.append(y.cpu())
        acc = correct / max(1,n)
        acc3 = correct_top3 / max(1,n)
        return acc, acc3, torch.cat(all_preds), torch.cat(all_tgts)

    def train(self):
        best = 0.0
        for ep in range(1, self.cfg.epochs+1):
            self.clip_branch.eval(); self.fusion.train()
            losses = []
            for batch in self.train_dl:
                loss, _ = self._step(batch, train=True)
                losses.append(loss)
            acc, acc3, _, _ = self._eval_epoch()
            print(f"[Ep {ep}] loss={np.mean(losses):.4f}  val@1={acc:.3f}  val@3={acc3:.3f}  alpha={self.fusion.alpha.item():.2f} beta={self.fusion.beta.item():.2f} T={self.clip_branch.logit_scale.exp().item():.2f}")
            if acc > best:
                best = acc
                self.save("best.pt")
        print("Training done. Best val@1:", best)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({
            'fusion': self.fusion.state_dict(),
            'clip': self.clip_branch.state_dict(),
            'meta': {
                'countries': COUNTRIES,
            }
        }, path)

# -----------------------------
# 7) Inference (fast path)
# -----------------------------

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
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
        ])

    @torch.no_grad()
    def predict_one(self, image_path: str, return_timing: bool = True) -> Dict[str, Any]:
        img = Image.open(image_path).convert('RGB')
        x = self.tf(img).unsqueeze(0).to(self.device, non_blocking=True)
        torch.cuda.synchronize() if self.device.type=="cuda" else None
        t0 = time.perf_counter()
        with torch.autocast(device_type=self.device.type if self.device.type!="mps" else 'cpu', enabled=(self.device.type=="cuda")):
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

# -----------------------------
# 8) Evaluation utilities
# -----------------------------

@torch.no_grad()
def evaluate_folder(pipeline: InferencePipeline, folder: str) -> Dict[str, Any]:
    y_true, y_pred = [], []
    latencies = []
    for cname in COUNTRIES:
        class_dir = os.path.join(folder, cname)
        if not os.path.isdir(class_dir):
            continue
        for fp in glob.glob(os.path.join(class_dir, '*')):
            if not os.path.isfile(fp):
                continue
            res = pipeline.predict_one(fp)
            y_true.append(NAME_TO_IDX[cname])
            y_pred.append(res['topk_indices'][0])
            if res['latency_s'] is not None:
                latencies.append(res['latency_s'])
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    acc = float((y_true==y_pred).mean()) if len(y_true)>0 else 0.0
    # confusion matrix
    K = NUM_CLASSES
    cm = np.zeros((K,K), dtype=int)
    for t,p in zip(y_true, y_pred):
        cm[t,p]+=1
    return {
        'accuracy': acc,
        'confusion_matrix': cm.tolist(),
        'mean_latency_ms': (np.mean(latencies)*1000.0) if latencies else None,
    }

# -----------------------------
# 9) CLI helpers
# -----------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd')
    
    p_train = sub.add_parser('train')
    p_train.add_argument('--root', type=str, required=True)
    p_train.add_argument('--image_size', type=int, default=336)
    p_train.add_argument('--batch_size', type=int, default=16)
    p_train.add_argument('--epochs', type=int, default=10)
    p_train.add_argument('--lr', type=float, default=1e-3)
    p_train.add_argument('--num_workers', type=int, default=4)

    p_eval = sub.add_parser('eval')
    p_eval.add_argument('--root', type=str, required=True)
    p_eval.add_argument('--weights', type=str, default='best.pt')
    p_eval.add_argument('--image_size', type=int, default=336)
    p_eval.add_argument('--no_ocr', action='store_true')

    p_pred = sub.add_parser('predict')
    p_pred.add_argument('--image', type=str, required=True)
    p_pred.add_argument('--weights', type=str, default='best.pt')
    p_pred.add_argument('--image_size', type=int, default=336)
    p_pred.add_argument('--no_ocr', action='store_true')

    args = p.parse_args()
    if args.cmd == 'train':
        cfg = TrainConfig(root=args.root, image_size=args.image_size, batch_size=args.batch_size, epochs=args.epochs, lr=args.lr, num_workers=args.num_workers)
        tr = Trainer(cfg)
        tr.train()
    elif args.cmd == 'eval':
        pipe = InferencePipeline(InferConfig(weights=args.weights, image_size=args.image_size, use_ocr=not args.no_ocr))
        res = evaluate_folder(pipe, args.root)
        print(json.dumps(res, indent=2))
    elif args.cmd == 'predict':
        pipe = InferencePipeline(InferConfig(weights=args.weights, image_size=args.image_size, use_ocr=not args.no_ocr))
        out = pipe.predict_one(args.image)
        print(json.dumps(out, indent=2))
    else:
        p.print_help()
