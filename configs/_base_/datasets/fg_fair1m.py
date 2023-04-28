# dataset settings
dataset_type = 'FGFAIR1MDataset'
data_root = '/home/chip/datasets/FAIR1M/mmrotate/split_ss_fair1m/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
angle_version = 'le90'
train_pipeline = [
    dict(type='LoadDFTImgFromFile'),
    dict(type='LoadFgAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='FgDefaultFormatBundle'),
    dict(type='Collect',keys=['img', 'gt_bboxes', 'gt_labels', 'gt_dom_labels'])
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
        ann_file=data_root + 'train/annfiles/',
        img_prefix=data_root + 'train/images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val/annfiles/',
        img_prefix=data_root + 'val/images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test/1_0/images/',
        img_prefix=data_root + 'test/1_0/images/',
        pipeline=test_pipeline))
