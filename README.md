# Generlized-OutlierExposure

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)


![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)


This repository is the official implementation of Generalized Outlier Exposure. A part of code has been based on the public code of
[Outlier Exposure](https://github.com/hendrycks/outlier-exposure), [SC-OOD](https://github.com/jingkang50/ICCV21_SCOOD), [Unknown Detection](https://github.com/daintlab/unknown-detection-benchmarks), [Mixup](https://github.com/facebookresearch/mixup-cifar10).

<img align="center" src="./fig/visualize.png" width="700">

### Environment

* Python >= 3.6

* Pytorch >= 1.9

* CUDA >= 10.2

## In-distribution Dataset
* CIFAR10, CIFAR100

## Outlier Dataset for train

Unlike the original [OE paper](https://arxiv.org/abs/1812.04606), which remove some examples from the outlier dataset, we used the outlier dataset as it is. 

* [**80 Million Tiny Images**](http://www.archive.org/download/80-million-tiny-images-2-of-2/tiny_images.bin)

## Outlier Datasets for test

SC-OOD dataset can be downloaded by the following link: [SC-OOD dataset download](https://drive.google.com/file/d/1cbLXZ39xnJjxXnDM7g2KODHIjE0Qj4gu/view).
* [SC-OOD dataset](https://github.com/jingkang50/ICCV21_SCOOD)
* Blobs
* Gaussian

## Train a model

```Python
python main.py --dataset cifar100 --model res34 --gpu-id 0 --trial 01 --filtered_num 20 ----estimation-func msp --strategy static --save-path ./save-path/
```

## Evaluate a model

```Python
python test.py --dataset cifar100 --model res34 --gpu-id 0 --save-path ./save-path/
```

### Evaluation metrics
* ACC
* AURC
* FRR at 95% TPR
* AUROC
* AUPR

## Results
<img align="center" src="./fig/main_results.png" width="700">
