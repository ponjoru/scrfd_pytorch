from scrfd.scrfd import SCRFD
from scrfd.backbone import MobileNetV1
from scrfd.resnet import ResNetV1e
from scrfd.neck import PAFPN
from scrfd.head import SCRFDHead


def init_scrfd_500m_model(load_from=None, device='cpu', use_kps=True):
    # init scrfd500m
    backbone = MobileNetV1(
        block_cfg={
            'stage_blocks': [2, 3, 2, 6],
            'stage_planes': [16, 16, 40, 72, 152, 288],
        }
    )
    neck = PAFPN(
        in_channels=[40, 72, 152, 288],
        out_channels=16,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=3,
    )
    bbox_head = SCRFDHead(
        num_classes=1,
        in_channels=16,
        stacked_convs=2,
        dw_conv=True,
        num_anchors=[2, 2, 2],
        strides=[8, 16, 32],
        use_kps=use_kps,
    )
    model = SCRFD(backbone=backbone, neck=neck, bbox_head=bbox_head)
    model.to(device)

    if load_from:
        model.load_from_checkpoint(load_from, strict=False)

    return model


def init_scrfd_2_5g_model(load_from=None, device='cpu', use_kps=True):
    # init scrfd2.5g
    backbone = ResNetV1e(
        depth=0,
        block_cfg=dict(
            block='BasicBlock',
            stage_blocks=(3, 5, 3, 2),
            stage_planes=[24, 48, 48, 80]),
        base_channels=24,
        num_stages=4,
    )
    neck = PAFPN(
        in_channels=[24, 48, 48, 80],
        out_channels=24,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=3,
    )
    bbox_head = SCRFDHead(
        num_classes=1,
        in_channels=24,
        stacked_convs=2,
        feat_channels=64,
        dw_conv=False,
        num_anchors=[2, 2, 2],
        strides=[8, 16, 32],
        use_kps=use_kps,
        use_scale=True,
    )
    model = SCRFD(backbone=backbone, neck=neck, bbox_head=bbox_head)
    model.to(device)

    if load_from:
        model.load_from_checkpoint(load_from, strict=False)

    return model


def init_scrfd_10g_model(load_from, device='cpu', use_kps=True):
    # init scrfd10g
    backbone = ResNetV1e(
        depth=0,
        block_cfg={
            'block': 'BasicBlock',
            'stage_blocks': (3, 4, 2, 3),
            'stage_planes': [56, 88, 88, 224],
        },
        base_channels=56,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
    )
    neck = PAFPN(
        in_channels=[56, 88, 88, 224],
        out_channels=56,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=3,
    )
    bbox_head = SCRFDHead(
        num_classes=1,
        in_channels=56,
        stacked_convs=3,
        feat_channels=80,
        dw_conv=False,
        num_anchors=[2, 2, 2],
        strides=[8, 16, 32],
        use_scale=True,
        use_kps=use_kps,
    )
    model = SCRFD(backbone=backbone, neck=neck, bbox_head=bbox_head)
    model.to(device)

    if load_from:
        model.load_from_checkpoint(load_from, strict=False)

    return model


def init_scrfd_34g_model(load_from, device='cpu', use_kps=True):
    print('WARNING: the original SCRFD34G model has strides shared head, and the current one has not')
    # init scrfd34g
    backbone = ResNetV1e(
        depth=0,
        block_cfg=dict(
            block='Bottleneck',
            stage_blocks=(17, 16, 2, 8),
            stage_planes=[56, 56, 144, 184]),
        base_channels=56,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
    )
    neck = PAFPN(
        in_channels=[224, 224, 576, 736],
        out_channels=128,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=3,
    )
    bbox_head = SCRFDHead(
        num_classes=1,
        in_channels=128,
        stacked_convs=2,
        feat_channels=256,
        dw_conv=False,
        num_anchors=[2, 2, 2],
        strides=[8, 16, 32],
        use_scale=True,
        use_kps=use_kps,
    )
    model = SCRFD(backbone=backbone, neck=neck, bbox_head=bbox_head)
    model.to(device)

    if load_from:
        model.load_from_checkpoint(load_from, strict=False)

    return model
