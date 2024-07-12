# MMYOLO

## Introduction

MMYOLO is a collection of YOLO model which is implemented by MMLabs with MMEngine and PyTorch. We use this repository to train and evaluate YOLO models across different datasets.

## Models

We provide the following YOLO models that are trained on different datasets.

| Model           | Version |                      Dataset Version                       | mAP50 |  mAP  | Config                                                                                                   | Weights                                                                                    |
| --------------- | :-----: | :--------------------------------------------------------: | :---: | :---: | -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| YOLOX-s Vehicle |  v1.0   | [v2.5](https://github.com/widyamsib/ppe-dataset/tree/v2.5) | 45.0  | 20.0  | [yolox_s_fast_8xb32-300e-rtmdet-hyp_coco.py](./configs/yolox/yolox_s_fast_8xb32-300e-rtmdet-hyp_coco.py) | [ONNX](https://github.com/widyamsib/mmyolo-msib/releases/download/wika-v1.0/epoch_15.onnx) |
| YOLOX-s Wika-Coco |  v2.0   | [v2.7](https://github.com/widyamsib/ppe-dataset/tree/v2.7)| 79.3  | 20.0  | [yolox_s_fast_1xb12-40e-rtmdet-hyp_wika2.py](./configs/yolox/yolox_s_fast_1xb12-40e-rtmdet-hyp_wika2.py) | [Weights](https://github.com/widyamsib/mmyolo-msib/releases/download/wika-v2.0/wika-coco.zip) |

## Getting Started

Please provide documentation.

## Training

Please provide documentation.

## Evaluation

Please provide documentation.