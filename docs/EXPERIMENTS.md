## Pretrained
```
python -m countrydet.main predict --image src/UZB_12.jpg --image_size 224 --weights "none" > logs/untrained_metrics.json
```

{
  "topk_indices": [
    23,
    22,
    8,
    0,
    13
  ],
  "topk_probs": [
    0.3186055123806,
    0.042356349527835846,
    0.03431519865989685,
    0.03352043405175209,
    0.03328753635287285
  ],
  "topk_labels": [
    "UZB",
    "USA",
    "EST",
    "BEL",
    "ITA"
  ],
  "latency_s": 151.470315487
}

## Fine-tuned
Let's train before!
```
python -m countrydet.main train --root dataset/train_val --epochs 10 --batch_size 2 --image_size 224 --lr 1e-4
```