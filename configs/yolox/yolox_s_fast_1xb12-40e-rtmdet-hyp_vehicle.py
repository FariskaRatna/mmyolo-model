_base_ = './yolox_s_fast_8xb32-300e-rtmdet-hyp_coco.py'

data_root = '/home/msib/datasets/coco_dataset/'
# class_name = ('truck', 'car', 'bus', 'bicycle', 'motorcycle')
# num_classes = len(class_name)
# metainfo = dict(classes=class_name, palette=[(20, 220, 60)])
import colorsys

class_name = ('bicycle', 'car', 'motorcycle', 'bus', 'truck') 

num_classes = len(class_name)

def generate_palette(num_classes):
    num_colors = max(num_classes, 1)
    hsv_tuples = [(x / num_colors, 1., 1.) for x in range(num_colors)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    palette = [(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)) for color in colors]
    return palette

palette = generate_palette(num_classes)

metainfo = dict(classes=class_name, palette=palette)

num_last_epochs = 5

max_epochs = 20
train_batch_size_per_gpu = 16
train_num_workers = 4

load_from = 'https://download.openmmlab.com/mmyolo/v0/yolox/yolox_s_fast_8xb32-300e-rtmdet-hyp_coco/yolox_s_fast_8xb32-300e-rtmdet-hyp_coco_20230210_134645-3a8dfbd7.pth'  # noqa

model = dict(
    backbone=dict(frozen_stages=4),
    bbox_head=dict(head_module=dict(num_classes=num_classes)))

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train_vehicle2024.json',
        data_prefix=dict(img='train_vehicle2024/')))

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='val_vehicle2024.json',
        data_prefix=dict(img='val_vehicle2024/')))

test_dataloader = val_dataloader

param_scheduler = [
    dict(
        # use quadratic formula to warm up 3 epochs
        # and lr is updated by iteration
        # TODO: fix default scope in get function
        type='mmdet.QuadraticWarmupLR',
        by_epoch=True,
        begin=0,
        end=3,
        convert_to_iter_based=True),
    dict(
        # use cosine lr from 5 to 35 epoch
        type='CosineAnnealingLR',
        eta_min=_base_.base_lr * 0.05,
        begin=5,
        T_max=max_epochs - num_last_epochs,
        end=max_epochs - num_last_epochs,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        # use fixed lr during last num_last_epochs epochs
        type='ConstantLR',
        by_epoch=True,
        factor=1,
        begin=max_epochs - num_last_epochs,
        end=max_epochs,
    )
]

_base_.custom_hooks[0].num_last_epochs = num_last_epochs

val_evaluator = dict(ann_file=data_root + 'val_vehicle2024.json')
test_evaluator = val_evaluator

default_hooks = dict(
    checkpoint=dict(interval=5, max_keep_ckpts=2, save_best='auto'),
    logger=dict(type='LoggerHook', interval=5))
train_cfg = dict(max_epochs=max_epochs, val_interval=5)

visualizer = dict(
    vis_backends=[
        dict(type="LocalVisBackend"),
        dict(
            type="MLflowVisBackend",
            save_dir="temp_dir",
            tracking_uri="http://localhost:5000",
            )
    ]
)

# visualizer = dict(vis_backends = [dict(type='LocalVisBackend'), dict(type='WandbVisBackend')]) # noqa
