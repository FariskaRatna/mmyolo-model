# MMYOLO

## Introduction

MMYOLO is a collection of YOLO model which is implemented by MMLabs with MMEngine and PyTorch. We use this repository to train and evaluate YOLO models across different datasets.

## Models

We provide the following YOLO models that are trained on different datasets.

| Model           | Version |                      Dataset Version                       | mAP50 |  mAP  | Config                                                                                                   | Weights                                                                                    |
| --------------- | :-----: | :--------------------------------------------------------: | :---: | :---: | -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| YOLOX-s Vehicle |  v1.0   | [v2.5](https://github.com/widyamsib/ppe-dataset/tree/v2.5) | 45.0  | 20.0  | [yolox_s_fast_8xb32-300e-rtmdet-hyp_coco.py](./configs/yolox/yolox_s_fast_8xb32-300e-rtmdet-hyp_coco.py) | [ONNX](https://github.com/widyamsib/mmyolo-msib/releases/download/wika-v1.0/epoch_15.onnx) |
| YOLOX-s Wika-Coco |  v2.0   | [v2.7](https://github.com/widyamsib/ppe-dataset/tree/v2.7)| 80.3  | 20.0  | [yolox_s_fast_1xb12-40e-rtmdet-hyp_wika2.py](./configs/yolox/yolox_s_fast_1xb12-40e-rtmdet-hyp_wika2.py) | [Weights](https://github.com/widyamsib/mmyolo-msib/releases/download/wika-v2.0/wika-coco.zip) |

## Getting Started

The first thing that should do is install MMYOLO and depedency libraries using the following commands.

```
pip install -U openmim
mim install -r requirements/mminstall.txt
mim install -v -e
```

## Training

You can training the model using the following commands.

```
python tools/train.py {config_models}
```

For `{config_models}` we can build the config file inside the config folder and choose the YOLO model that we want to use for the training process. For example,if we try to train the WIKA-UNJANI dataset using YOLOX, we create a `yolox_s_fast_1xb12-40e-rtmdet-hyp_wika.py` config file in the `configs\yolox` folder and write down the configuration of the model. So, the command to training the model is.

```
python tools/train.py configs/yolox/yolox_s_fast_1xb12-40e-rtmdet-hyp_wika.py
```

After running the command, `work_dirs/yolox_s_fast_1xb12-40e-rtmdet-hyp_wika` folder will be automatically generated, the checkpoint file and the training config file will be saved in this folder.

### Training is resumed after interruption

If you stop training, you can add `--resume` to the end of the training command and the program will aoutomatically resume training with the latest weights file from `work_dirs`.

```
python tools/train.py configs/yolox/yolox_s_fast_1xb12-40e-rtmdet-hyp_wika.py --resume
```

## Evaluation

