## Requirements

- python==3.9.18
- torch==1.12.1+cu102
- transformers==4.35.2
- numpy==1.26.2
- pillow==10.1.0
- tokenizers==0.15.0
- rouge==1.0.1
- nltk==3.8.1
- bert-score==0.3.13
- krippendorff==0.6.1

## Overview

Please unzip dataset_mmss.zip and follow the following file structure.

```
root
├── dataset_mmss
│   ├── train
│   │   ├── texts.json
│   │   └── images
│   │       ├── 1.jpg
│   │       ├── 2.jpg
│   │       ├── ...
│   │       └── 62000.jpg
│   ├── val
│   │   ├── texts.json
│   │   └── images
│   │       ├── 1.jpg
│   │       ├── 2.jpg
│   │       ├── ...
│   │       └── 2000.jpg
│   └── test
│       ├── texts.json
│       └── images
│           ├── 1.jpg
│           ├── 2.jpg
│           ├── ...
│           └── 2000.jpg
├── dataset.py
├── model.py
├── scorer.py
├── test.py
└── train.py
```

## How to train

```
CUDA_VISIBLE_DEVICES=0 python train.py --seed 1 --train_dataset_proportion 1.0 --val_dataset_proportion 1.0 --train_batch_size 16 --evaluate_batch_size 16 --kld_loss_weight 0.075 --positive_loss_weight 2.0 --negative_loss_weight 2.0 --eval --num_epochs 50 --learning_rate 5e-6 --num_beams 10
```

## How to infer

```
CUDA_VISIBLE_DEVICES=0 python test.py --seed 1 --test_dataset_proportion 1.0 --evaluate_batch_size 16 --num_beams 10
```
