"""
-----------------------------
Evaluation utilities
-----------------------------
"""

import os
import glob
import numpy as np
from typing import Any
from tqdm import tqdm

import torch

from countrydet.engine.cfg import COUNTRIES, NAME_TO_IDX, NUM_CLASSES
from countrydet.engine.inference import InferencePipeline


@torch.no_grad()
def evaluate_folder(pipeline: InferencePipeline, folder: str) -> dict[str, Any]:
    y_true, y_pred = [], []
    latencies = []
    for cname in COUNTRIES:
        print(cname)
        class_dir = os.path.join(folder, cname)
        if not os.path.isdir(class_dir):
            continue
        for fp in tqdm(glob.glob(os.path.join(class_dir, '*'))):
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