# RetinaFace in PyTorch (Running Face-detection on Images)

A [PyTorch](https://pytorch.org/) implementation of [RetinaFace: Single-stage Dense Face Localisation in the Wild](https://arxiv.org/abs/1905.00641). Model size only 1.7M, when Retinaface use mobilenet0.25 as backbone net. The official code in Mxnet can be found [here](https://github.com/deepinsight/insightface/tree/master/RetinaFace).

## WiderFace Val Performance in single scale When using Mobilenet0.25 as backbone net.
| Style | easy | medium | hard |
|:-|:-:|:-:|:-:|
| Pytorch (same parameter with Mxnet) | 86.85 % | 85.84% | 79.69% |
| Pytorch (original image scale) | 90.58 % | 87.94% | 73.96% |
| Original Mxnet | - | - | 79.1% |

## FDDB Performance When using Mobilenet0.25 as backbone net.
| Dataset | performace |
|:-|:-:|
| FDDB(pytorch) | 97.93% |

## Installation
##### Clone and install
1. git clone https://github.com/vasu-dev/Pytorch_Retinaface.git

2. Pytorch version 1.1.0+ and torchvision 0.3.0+ are needed.

3. Codes are based on Python 3


## TESTING ON IMAGE
1. python face_detection.py
