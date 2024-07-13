from mmdet.apis import init_detector, inference_detector

config_file = 'configs/rtmdet/rotated/rtmdet-r_l_syncbn_fast_2xb4-aug-100e_dota.py'
checkpoint_file = 'rtmdet-r_l_syncbn_fast_2xb4-aug-100e_dota_20230224_124735-ed4ea966.pth'
model = init_detector(config_file, checkpoint_file, device='cpu')
inference_detector(model, 'demo/P0006.jpg')