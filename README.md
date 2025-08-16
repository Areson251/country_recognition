# COUNTRYDET
![Pipeline scheme](src/scheme.png)

# Installation
docker
```
pip install numpy torch pillow easyocr open_clip_torch rapidfuzz
```

# Prepare data
dataset_orig -> split in dataset folder
TRAIN_VAL_COUNT = 90 - samples count in train+val
TEST_COUNT = 10 - samples count in test
```
python countrydet/dataset/split_imgs.py
```

