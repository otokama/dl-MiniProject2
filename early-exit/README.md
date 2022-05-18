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

1. Model training:
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
2. Early exit models:
    - ResNet-20: ```--arch=resnet20_cifar_earlyexit```
    - ResNet-32: ```--arch=resnet32_cifar_earlyexit```
    - ResNet-56: ```--arch=resnet56_cifar_earlyexit```
    - ResNet-110: ```--arch=resnet110_cifar_earlyexit```
3. Example command for training ResNet-56:
```
python distiller-4-bit-qat/examples/classifier_compression/compress_classifier.py 
    --arch=resnet56_cifar_earlyexit
    --epochs=300 -b 128 --lr 0.1 data/ --weight-decay 0.0001 --momentum 0.9 
    --earlyexit_thresholds 1.2 --earlyexit_lossweights 0.1
```

Note: Distiller supports training on CIFAR-10 out of the box. You need to modify train and test dataloaders in ```apputils/data_loaders.py``` and ```NUM_CLASSES``` in ```models/cifar10/resnet_cifar_earlyexit.py``` to support training on CIFAR-100.
