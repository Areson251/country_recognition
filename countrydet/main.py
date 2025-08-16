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
import json

import torch

# Optional (speed/quality): set torch.backends flags if on GPU
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


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
