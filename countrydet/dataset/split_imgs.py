import os
import shutil
import random
from tqdm import tqdm


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "dataset_orig")
DST_DIR = os.path.join(BASE_DIR, "dataset")

TRAIN_VAL_DIR = os.path.join(DST_DIR, "train_val")
TEST_DIR = os.path.join(DST_DIR, "test")
TRAIN_VAL_COUNT = 90
TEST_COUNT = 10

os.makedirs(TRAIN_VAL_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

random.seed(42)

countries = [
    d for d in os.listdir(SRC_DIR)
    if os.path.isdir(os.path.join(SRC_DIR, d)) and not d.startswith(".")
]

for country in tqdm(countries):
    country_path = os.path.join(SRC_DIR, country)
    if not os.path.isdir(country_path):
        continue

    train_val_country = os.path.join(TRAIN_VAL_DIR, country)
    test_country = os.path.join(TEST_DIR, country)
    os.makedirs(train_val_country, exist_ok=True)
    os.makedirs(test_country, exist_ok=True)

    images = [f for f in os.listdir(country_path) if os.path.isfile(os.path.join(country_path, f))]
    random.shuffle(images)

    assert len(images) >= TRAIN_VAL_COUNT + TEST_COUNT, \
        f"Not enough {len(images)} images for {TRAIN_VAL_COUNT}+{TEST_COUNT} split"

    train_val_imgs = images[:TRAIN_VAL_COUNT]
    test_imgs = images[TRAIN_VAL_COUNT:TRAIN_VAL_COUNT + TEST_COUNT]

    for img in train_val_imgs:
        shutil.copy(os.path.join(country_path, img), os.path.join(train_val_country, img))

    for img in test_imgs:
        shutil.copy(os.path.join(country_path, img), os.path.join(test_country, img))

print(f"Done: train_val={TRAIN_VAL_COUNT}, test={TEST_COUNT} for each folder")