# Baseline model training


## Environment

1. Create conda environment
```
conda env create -f environment.yml
```

2. Activate conda environment
```
conda activate baseline-env
```

## Usage

1. Model training:
```
python baseline_model.py 
--lr [learning rate] 
--epoch [number of epoch] 
--model [baseline model] 
--dataset [cifar10 or cifar100]
```

2. Baseline models:
- ResNet-20:  ```--model 0  ```
- ResNet-32:  ```--model 1 ```
- ResNet-56:  ```--model 2```
- ResNet-110:  ```--model 3```
