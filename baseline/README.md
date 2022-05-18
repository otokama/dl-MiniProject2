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

3. Example command for training ResNet-32 for 200 epochs using default learning rate (0.1) and dataset (CIFAR10):
```
python baseline_model.py --epoch 200 --model 1
```
