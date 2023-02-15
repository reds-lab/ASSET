# ASSET: Robust Backdoor Data Detection Across a Multiplicity of Deep Learning Paradigms

![Python 3.6](https://img.shields.io/badge/python-3.6-DodgerBlue.svg?style=plastic)
![Pytorch 1.10.1](https://img.shields.io/badge/pytorch-1.10.1-DodgerBlue.svg?style=plastic)
![CUDA 11.0](https://img.shields.io/badge/cuda-11.0-DodgerBlue.svg?style=plastic)

<div align=center>
<img align="right" width="350px" style="margin-left: 25px; margin-top: 8px" src="https://user-images.githubusercontent.com/77789132/218630276-20c8ee0c-61dd-4518-b4fb-72943518596f.gif">
</div>


This repository is the official implementation of the paper "[ASSET: Robust Backdoor Data Detection Across a Multiplicity of Deep Learning Paradigms](https://www.yi-zeng.com/)." We find that existing detection methods cannot be applied or suffer limited performance for Self-Supervised Learning and transfer learning; even for the widely studied end-to-end supervised learning setting, there is still large room to improve detection in terms of their robustness to variations in poison ratio and attack designs.

To address this problem...actively introduce diffrent model behaviors...

# Features
<div align=center>
<img src="https://user-images.githubusercontent.com/77789132/218583421-1184b200-5dd0-418a-82a7-15754704fc2f.png">
</div>

In the past, the detection of backdoor data was primarily researched within the framework of end-to-end supervised learning (SL). However, in recent years, the use of self-supervised learning (SSL) and transfer learning (TL) has become increasingly popular due to their reduced requirement for labeled data. It has also been shown that successful backdoor attacks can be carried out in these novel settings. Wepropose a new detection method called Active Separation via Offset (ASSET), which actively induces different model behaviors between the backdoor and clean samples to promote their separation. ASSET enables stable defense under different learning paradigms. 

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

