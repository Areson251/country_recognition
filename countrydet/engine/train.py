"""
-----------------------------
Training (fine‑tune fusion)
-----------------------------
"""

import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from countrydet.dataset.dataset import DocDataset
from countrydet.engine.cfg import COUNTRIES, NUM_CLASSES
from countrydet.models.feature_extractor import CLIPScorer
from countrydet.models.fusion import FusionHead

# Optional (speed/quality): set torch.backends flags if on GPU
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


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
    def __init__(self, cfg: TrainConfig, log_dir: str = "runs/fusion_train"):
        self.cfg = cfg
        self.device = torch.device('cuda')
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_ds = DocDataset(cfg.root, split="train", image_size=cfg.image_size)
        self.val_ds = DocDataset(cfg.root, split="val", image_size=cfg.image_size)

        self.train_dl = DataLoader(self.train_ds, 
                                   batch_size=cfg.batch_size, 
                                   shuffle=True, 
                                   num_workers=cfg.num_workers, 
                                   pin_memory=True)
        self.val_dl = DataLoader(self.val_ds, 
                                 batch_size=cfg.batch_size, 
                                 shuffle=False, 
                                 num_workers=cfg.num_workers, 
                                 pin_memory=True)

        self.clip_branch = CLIPScorer(image_size=cfg.image_size).to(self.device)
        self.fusion = FusionHead(NUM_CLASSES).to(self.device)
        self.opt = torch.optim.AdamW(list(self.fusion.parameters()) + [self.clip_branch.logit_scale], 
                                                                        lr=cfg.lr, weight_decay=1e-4)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type=="cuda" and cfg.mixed_precision))

        self.writer = SummaryWriter(log_dir=log_dir)
        self.global_step = 0

    def _step(self, batch, train: bool):
        x, y, _ = batch
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        with torch.autocast(device_type=self.device.type if self.device.type!="mps" 
                            else 'cpu', enabled=(self.device.type=="cuda" and self.cfg.mixed_precision)):
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
            self.writer.add_scalar("train/loss_step", loss.item(), self.global_step)
            self.global_step += 1

        return loss.detach().item(), logits.detach()

    @torch.no_grad()
    def _eval_epoch(self, epoch: int):
        self.clip_branch.eval(); self.fusion.eval()
        n, correct, correct_top3 = 0, 0, 0
        all_preds, all_tgts = [], []
        for x, y, _ in self.val_dl:
            x = x.to(self.device)
            y = y.to(self.device)
            logits = self.fusion(self.clip_branch(x), 
                                    torch.zeros((x.size(0), NUM_CLASSES), 
                                    device=self.device))
            top1 = logits.argmax(dim=-1)
            top3 = logits.topk(3, dim=-1).indices
            correct += (top1==y).sum().item()
            correct_top3 += (top3==y.unsqueeze(1)).any(dim=1).sum().item()
            n += x.size(0)
            all_preds.append(top1.cpu())
            all_tgts.append(y.cpu())
        acc = correct / max(1,n)
        acc3 = correct_top3 / max(1,n)

        self.writer.add_scalar("val/acc@1", acc, epoch)
        self.writer.add_scalar("val/acc@3", acc3, epoch)
        self.writer.add_scalar("params/alpha", self.fusion.alpha.item(), epoch)
        self.writer.add_scalar("params/beta", self.fusion.beta.item(), epoch)
        self.writer.add_scalar("params/T", self.clip_branch.logit_scale.exp().item(), epoch)

        return acc, acc3, torch.cat(all_preds), torch.cat(all_tgts)

    def train(self):
        best = 0.0
        best_epoch = -1
        for ep in range(1, self.cfg.epochs+1):
            self.clip_branch.eval(); self.fusion.train()
            losses = []
            for batch in self.train_dl:
                loss, _ = self._step(batch, train=True)
                losses.append(loss)
            
            mean_loss = np.mean(losses)
            self.writer.add_scalar("train/loss_epoch", mean_loss, ep)

            acc, acc3, _, _ = self._eval_epoch(ep)
            print(f"[Ep {ep}] loss={mean_loss:.4f}  val@1={acc:.3f}  val@3={acc3:.3f}  "
                  f"alpha={self.fusion.alpha.item():.2f} beta={self.fusion.beta.item():.2f} "
                  f"T={self.clip_branch.logit_scale.exp().item():.2f}")
            if acc > best:
                best = acc
                best_epoch = ep
                self.save("best.pt")
        print(f"Training done. Best val@1: {best} at epoch: {best_epoch}")
        self.writer.close()

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({
            'fusion': self.fusion.state_dict(),
            'clip': self.clip_branch.state_dict(),
            'meta': {
                'countries': COUNTRIES,
            }
        }, path)