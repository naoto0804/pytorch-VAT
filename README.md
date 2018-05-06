# pytorch-VAT

This is an unofficial pytorch implementation of a paper, Distributional Smoothing with Virtual Adversarial Training [Miyato+, ICLR2016].

Please note that this is an ongoing project and I cannot fully reproduce the results currently.


## Requirements
- Python 3.5+
- PyTorch 0.4
- TorchVision
- click


## Usage

### Train classifier with only labeled data (100 images)
```
CUDA_VISIBLE_DEVICES=<gpu_id> python train_baseline.py --n_label 100
```
Error rate: about 30%

### Train classifier with mixture of labeled and unlabeled data (100 images)
```
CUDA_VISIBLE_DEVICES=<gpu_id> python train_baseline.py --n_label 100
```
Error rate: about 2%

## References
- [1]: T. Miyato et al. "Distributional Smoothing with Virtual Adversarial Training", in ICLR, 2016.
