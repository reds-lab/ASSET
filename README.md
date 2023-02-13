# ASSET: Robust Backdoor Data Detection Across a Multiplicity of Deep Learning Paradigms

![Python 3.6](https://img.shields.io/badge/python-3.6-DodgerBlue.svg?style=plastic)
![Pytorch 1.10.1](https://img.shields.io/badge/pytorch-1.10.1-DodgerBlue.svg?style=plastic)
![CUDA 11.0](https://img.shields.io/badge/cuda-11.0-DodgerBlue.svg?style=plastic)


This repository is the official implementation of the paper "[ASSET: Robust Backdoor Data Detection Across a Multiplicity of Deep Learning Paradigms](https://www.yi-zeng.com/)." We find that existing detection methods cannot be applied or suffer limited performance for Self-Supervised Learning and transfer learning; even for the widely studied end-to-end supervised learning setting, there is still large room to improve detection in terms of their robustness to variations in poison ratio and attack designs.

# Features
![table](https://user-images.githubusercontent.com/77789132/218352301-421a9fe1-70d4-469f-91e8-0e9da2bdc823.png)

# Requirements
+ Python >= 3.6
+ PyTorch >= 1.10.1
+ Torchvision >= 0.11.2
+ Imageio >= 2.9.0


# Usage & HOW-TO
<p align="justify">Use the ASSET_demo.ipynb
 notebook for a quick start of the ASSET defense (demonstrated on the CIFAR-10 dataset). The default setting running on the CIFAR-10 dataset and attack method is BadNets on ResNet-18.</p>
 
# Can you make it easier?

