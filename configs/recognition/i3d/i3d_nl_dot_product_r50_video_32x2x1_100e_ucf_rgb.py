_base_ = [
    '../../_base_/models/i3d_r50.py', '../../_base_/schedules/sgd_100e.py',
    '../../_base_/default_runtime.py'
]

# model settings
model = dict(
    backbone=dict(
        non_local=((0, 0, 0), (0, 1, 0, 1), (0, 1, 0, 1, 0, 1), (0, 0, 0)),
        non_local_cfg=dict(
            sub_sample=True,
            use_scale=False,
            norm_cfg=dict(type='BN3d', requires_grad=True),
            mode='dot_product')))

# dataset settings
dataset_type = 'VideoDataset_ucfcrime'
data_root = '/media/hkuit155/24d4ed16-ee67-4121-8359-66a09cede5e7/AbnormalDetection/DATASET/UCF-crime/Anomaly-Videos'
data_root_val = '/media/hkuit155/24d4ed16-ee67-4121-8359-66a09cede5e7/AbnormalDetection/DATASET/UCF-crime/Anomaly-Videos'
ann_file_train = '/media/hkuit155/24d4ed16-ee67-4121-8359-66a09cede5e7/AbnormalDetection/DATASET/UCF-crime/ucfcrime_train_split_1_videos.txt'
ann_file_val = '/media/hkuit155/24d4ed16-ee67-4121-8359-66a09cede5e7/AbnormalDetection/DATASET/UCF-crime/ucfcrime_val_split_1_videos.txt'
ann_file_test = '/media/hkuit155/24d4ed16-ee67-4121-8359-66a09cede5e7/AbnormalDetection/DATASET/UCF-crime/ucfcrime_val_split_1_videos.txt'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=32, frame_interval=50, num_clips=32),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 112)),
    dict(
        type='MultiScaleCrop',
        input_size=112,
        scales=(1, 0.8),
        random_crop=False,
        max_wh_scale_gap=0),
    dict(type='Resize', scale=(112, 112), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=50,
        num_clips=32,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 112)),
    dict(type='CenterCrop', crop_size=112),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=50,
        num_clips=32,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 112)),
    dict(type='ThreeCrop', crop_size=112),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=2,
    workers_per_gpu=0,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=test_pipeline))

# runtime settings
work_dir = './work_dirs/i3d_nl_dot_product_r50_video_32x2x1_100e_ucf_rgb/'
