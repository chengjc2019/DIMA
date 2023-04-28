# dataset settings
dataset_type = 'MARDataset'
data_root = '/home/chip/datasets/Mar20/'

angle_version = 'le90'
img_norm_cfg = dict(
    # mean=[21.55, 21.55, 21.55], std=[24.42, 24.42, 24.42], to_rgb=True)
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(800, 800)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 800),
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
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        classwise=True,
        ann_file=data_root + 'ImageSets/Main/train.txt',
        ann_subdir=data_root + 'Annotations/Oriented Bounding Boxes',
        img_subdir=data_root + 'JPEGImages/',
        img_prefix=data_root + 'JPEGImages/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classwise=True,
        ann_file=data_root + 'ImageSets/Main/test.txt',
        ann_subdir=data_root + 'Annotations/Oriented Bounding Boxes',
        img_subdir=data_root + 'JPEGImages/',
        img_prefix=data_root + 'JPEGImages/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classwise=True,
        ann_file=data_root + 'ImageSets/Main/test.txt',
        ann_subdir=data_root + 'Annotations/Oriented Bounding Boxes',
        img_subdir=data_root + 'JPEGImages/',
        img_prefix=data_root + 'JPEGImages/',
        pipeline=test_pipeline))
