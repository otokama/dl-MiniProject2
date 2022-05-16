# Early exit model training

## Environment

1. Create conda environment
```
conda env create -f environment.yml
```

2. Activate conda environment
```
conda activate distiller-env
```

## Usage

```
python distiller-4-bit-qat/examples/classifier_compression/compress_classifier.py 
    --arch [model] 
    --epochs [number of epoch] 
    -b [batch size] 
    --lr [learning rate] 
    --wd [weight decay] 
    --momentum [momentum] 
    DIR [directory of CIFAR10 dataset]
    --earlyexit_thresholds [early exit threshold]
    --earlyexit_lossweights [early exit loss weights]
```

Note: Distiller supports training on CIFAR-10 out of the box. You need to modify train and test dataloaders in ```apputils/data_loaders.py``` and ```NUM_CLASSES``` in ```models/cifar10/resnet_cifar_earlyexit.py``` to support training on CIFAR-100.