# pytorch-VAT

This is an unofficial pytorch implementation of a paper, Distributional Smoothing with Virtual Adversarial Training [Miyato+, ICLR2016].

Please note that this is an ongoing project and I cannot fully reproduce the results currently.


## Requirements
- Python 3.5+
- PyTorch 0.4
- TorchVision
- click


## Usage

These examples are for the MNIST to USPS experiment.

### Train classifier with only labeled data
```
CUDA_VISIBLE_DEVICES=<gpu_id> python train_baseline.py
```
Error rate: about 30%

### Train classifier with mixture of labeled and unlabeled data
```
CUDA_VISIBLE_DEVICES=<gpu_id> python train_baseline.py
```
Error rate: about 2%

## References
- [1]: T. Miyato et al. "Distributional Smoothing with Virtual Adversarial Training", in ICLR, 2016.