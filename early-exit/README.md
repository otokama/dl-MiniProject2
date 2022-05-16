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
--m [momentum] 
DIR [directory of CIFAR10 dataset]
--earlyexit_thresholds [early exit threshold]
--earlyexit_lossweights [early exit loss weights]
```

