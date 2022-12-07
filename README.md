# Table of contents
1. [Installation](#Installation)
2. [Sample Usage](##Sample-Usage)

## Installation
```bash
pip install -r requirements.txt
```


## Sample Usage 
### Classification using all features

Proposed Multi-CNN applied to raw data - superficial vs deep OP (Male) at region 1200cm<sup>-1</sup>

`python multicnn.py --dataset "./dataset/raw/1200_superficial_vs_deep_OP_M.csv" --seed 0 --threads 1 --lr 0.01 --train-batch 4 --valid-batch 1 --epochs 100 --checkpoint-hist 1 --channel pspline_aspls jbcd beads ria fabc adaptive_minmax goldindec;`

Baseline CNN applied to raw data - superficial vs deep OP (Male) at region 1200cm<sup>-1</sup>

`python cnnDefTrainkfold.py --dataset "./dataset/raw/1200_superficial_vs_deep_OP_M.csv" --seed 0 --threads 1 --lr 0.01 --train-batch 4 --valid-batch 1 --epochs 100 --checkpoint-hist 1;`

Baseline CNN applied to pre-processed data - superficial vs deep OP (Male) at region 1200cm<sup>-1</sup>

`python cnnDefTrainkfold.py --dataset "./dataset/preprocessed/1200_superficial_vs_deep_OP_M.csv" --seed 0 --threads 1 --lr 0.01 --train-batch 4 --valid-batch 1 --epochs 100 --checkpoint-hist 1;`

### Feature selection
Please run the proposed Multi-CNN on the classification task prior to the following experiments:

SVM applied to raw data - superficial vs deep OP (Male) at region 1200cm<sup>-1</sup> using top 5 features

`python svm_dt_reduced.py --method "svm" --dataset "./dataset/raw/1200_superficial_vs_deep_OP_M.csv" --num-feat 5 --seed 0 --threads 1 --lr 0.01 --train-batch 4 --valid-batch 1 --epochs 100 --checkpoint-hist 1`

Decision Tree applied to raw data - superficial vs deep OP (Male) at region 1200cm<sup>-1</sup> using top 5 features

`python svm_dt_reduced.py --method "dt" --dataset "./dataset/raw/1200_superficial_vs_deep_OP_M.csv" --num-feat 5 --seed 0 --threads 1 --lr 0.01 --train-batch 4 --valid-batch 1 --epochs 100 --checkpoint-hist 1`

ANN applied to raw data - superficial vs deep OP (Male) at region 1200cm<sup>-1</sup> using top 5 features

`python ann_reduced.py --dataset "./dataset/raw/1200_superficial_vs_deep_OP_M.csv" --num-feat 5 --neurons 22 --seed 0 --threads 1 --lr 0.01 --train-batch 4 --valid-batch 1 --epochs 100 --checkpoint-hist 1;`




