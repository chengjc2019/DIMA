# dataset settings
dataset_type = 'FAIR1MDataset'
data_root = '/home/chip/datasets/FAIR1M/mmrotate/split_ss_fair1m/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

angle_version = 'le90'
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(type='RRandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    train=dict(pipeline=train_pipeline, version=angle_version),
    val=dict(version=angle_version),
    test=dict(version=angle_version))

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=16,
    train=dict(
        type=dataset_type,
        ann_file='/home/chip/datasets/FAIR1M/mmrotate/split_ss_trainval/train/annfiles/',
        img_prefix='/home/chip/datasets/FAIR1M/mmrotate/split_ss_trainval/train/images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val/annfiles/',
        img_prefix=data_root + 'val/images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        # ann_file=data_root + 'test/1_0/images/',
        # img_prefix=data_root + 'test/1_0/images/',
        ann_file=
        '/home/chip//datasets/FAIR1M/raw/test/1_0/png/',
        img_prefix=
        '/home/chip//datasets/FAIR1M/raw/test/1_0/png/',
        pipeline=test_pipeline))
