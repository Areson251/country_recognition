"""-----------------------------
OCR branch (EasyOCR + heuristics)
-----------------------------
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
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from countrydet.engine.cfg import COUNTRY_LIST, NAME_TO_IDX, NUM_CLASSES, ALPHA3_MAP

try:
    import easyocr  # PyTorch‑based OCR
    from rapidfuzz import fuzz
except Exception as e:
    easyocr = None
    fuzz = None
    print("[WARN] easyocr or rapidfuzz not found. Install with: pip install easyocr rapidfuzz")


MRZ_ISSUING_RE = re.compile(r"[IPV]\<([A-Z]{3})")  # matches e.g., P<DEU


class OCRCountryScorer:
    def __init__(self, languages: list[str] = ["en"]):
        if easyocr is None:
            raise ImportError("easyocr is required for OCRCountryScorer")
        self.reader = easyocr.Reader(languages, gpu=torch.cuda.is_available())
        # Flatten alias dict to (country_idx, alias_text)
        self.alias_list: list[tuple[int, str]] = []
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
        # MRZ issuing state code
        m = MRZ_ISSUING_RE.search(text_full.replace(" ", ""))
        if m:
            code = m.group(1)
            country = ALPHA3_MAP.get(code)
            if country is not None:
                scores[NAME_TO_IDX[country]] += 2.0  # strong signal
        # Alias fuzzy matches
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