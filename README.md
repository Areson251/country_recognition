# COUNTRYDET
![Pipeline scheme](src/scheme.png)

# Installation
`build.sh` build docker image

`start.sh` start docker container

`into.sh` go into docker container

```
chmod +x build.sh start.sh into.sh
./build.sh 
./start.sh 
./into.sh 
```

# Prepare data
dataset_orig -> split in dataset folder
TRAIN_VAL_COUNT = 90 - samples count in train+val
TEST_COUNT = 10 - samples count in test
```
python countrydet/dataset/split_imgs.py
```

# Train

# Inference

# Results
[See more](docs/EXPERIMENTS.md)

# Contact Me

TODO: ссылочка на тг и можно мем какой-нибудь