_base_ = ['./redet_re50_refpn_1x_fair1m_le90.py']

data_root = '/home/chip/datasets/FAIR1M/mmrotate/split_ms_fair1m/'
dataset_type = 'FGFAIR1MDataset'
angle_version = 'le90'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
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
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_dom_labels'])
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
    samples_per_gpu=2,
    workers_per_gpu=16,
    train=dict(
        type=dataset_type,
        pipeline=train_pipeline,
        ann_file=data_root + 'train/annfiles/',
        img_prefix=data_root + 'train/images/'),
    val=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        ann_file=data_root + 'val/annfiles/',
        img_prefix=data_root + 'val/images/'),
    test=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        ann_file=data_root + 'test/images/',
        img_prefix=data_root + 'test/images/'))

model = dict(train_cfg=dict(rpn=dict(assigner=dict(gpu_assign_thr=200))))
