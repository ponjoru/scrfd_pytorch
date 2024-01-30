import torch
import torch.nn as nn

from torch import Tensor
from typing import List, Tuple

from .utils import ConvBnAct, DWConvBnAct


class Scale(nn.Module):
    """A learnable scale parameter.

    This layer scales the input by a learnable factor. It multiplies a
    learnable scale parameter of shape (1,) with input of any shape.

    Args:
        scale (float): Initial value of scale factor. Default: 1.0
    """

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class SCRFDStrideHead(nn.Module):
    def __init__(self, nc, in_ch, feat_ch, num_anchors, n_conv_layers=4, dw_conv=False, export=False, use_scale=False):
        super(SCRFDStrideHead, self).__init__()

        self.nc = nc
        self.dw_conv = dw_conv
        self.export = export
        self.use_scale = use_scale

        conv_layers = []
        chn = in_ch
        for i in range(n_conv_layers):
            conv_layers.append(self._get_conv_module(chn, feat_ch))
            chn = feat_ch
        self.conv_layers = nn.Sequential(*conv_layers)

        self.cls_logits = nn.Conv2d(feat_ch, nc * num_anchors, kernel_size=(3, 3), padding=1)    # 2
        self.bb_logits = nn.Conv2d(feat_ch, 4 * num_anchors, kernel_size=(3, 3), padding=1)                         # 8

        if self.use_scale:
            self.scale = Scale(1.0)
        # 4 * (na + 1) dfl
        # self.stride_reg[key] = nn.Conv2d(feat_ch, 4 * (self.reg_max + 1) * self.num_anchors[stride_idx], 3, padding=1)

    def forward(self, x):
        feat = self.conv_layers(x)
        cls_score = self.cls_logits(feat)
        bb_pred = self.bb_logits(feat)

        if self.use_scale:
            bb_pred = self.scale(bb_pred)

        if self.export:
            bs = cls_score.shape[0]
            cls_score = cls_score.permute(0, 2, 3, 1).reshape(bs, -1, self.nc).sigmoid()
            bb_pred = bb_pred.permute(0, 2, 3, 1).reshape(bs, -1, 4)

        return cls_score, bb_pred

    def _get_conv_module(self, in_ch, out_ch):
        if self.dw_conv:
            conv = DWConvBnAct(in_ch, out_ch, 3, stride=1, padding=1, with_norm=True, activation=nn.ReLU)
        else:
            conv = ConvBnAct(in_ch, out_ch, 3, stride=1, padding=1, with_norm=True, activation=nn.ReLU)
        return conv


class SCRFDStrideFaceHead(SCRFDStrideHead):
    def __init__(self, nc, in_ch, feat_ch, num_anchors, n_conv_layers, dw_conv=False, export=False, use_scale=False):
        super(SCRFDStrideFaceHead, self).__init__(nc, in_ch, feat_ch, num_anchors, n_conv_layers, dw_conv, export, use_scale)

        self.nk = 5    # number of key points
        self.kp_logits = nn.Conv2d(feat_ch, 2 * self.nk * num_anchors, kernel_size=(3, 3), padding=1)  # 20

    def forward(self, x):
        feat = self.conv_layers(x)
        cls_score = self.cls_logits(feat)
        bb_pred = self.bb_logits(feat)
        kp_pred = self.kp_logits(feat)

        if self.use_scale:
            bb_pred = self.scale(bb_pred)

        if self.export:
            bs = cls_score.shape[0]
            cls_score = cls_score.permute(0, 2, 3, 1).reshape(bs, -1, self.nc).sigmoid()
            bb_pred = bb_pred.permute(0, 2, 3, 1).reshape(bs, -1, 4)
            kp_pred = kp_pred.permute(0, 2, 3, 1).reshape(bs, -1, self.nk)

        return cls_score, bb_pred, kp_pred


class SCRFDHead(nn.Module):
    def __init__(
        self,
        num_classes,
        in_channels,
        num_anchors,
        stacked_convs=4,
        feat_channels=64,
        strides=(8, 16, 32),
        dw_conv=False,
        use_scale=False,
        use_kps=True,
    ):
        super(SCRFDHead, self).__init__()
        self.strides = strides
        self.num_classes = num_classes
        self.use_kps = use_kps

        self.stride_heads = nn.ModuleList()
        for na, stride in zip(num_anchors, strides):
            if use_kps:
                head = SCRFDStrideFaceHead(
                    nc=num_classes,
                    in_ch=in_channels,
                    feat_ch=feat_channels,
                    num_anchors=na,
                    n_conv_layers=stacked_convs,
                    dw_conv=dw_conv,
                    use_scale=use_scale,
                )
            else:
                head = SCRFDStrideHead(
                    nc=num_classes,
                    in_ch=in_channels,
                    feat_ch=feat_channels,
                    num_anchors=na,
                    n_conv_layers=stacked_convs,
                    dw_conv=dw_conv,
                    use_scale=use_scale,
                )
            self.stride_heads.append(head)

    def _forward_kps(self, features: List[Tensor]):
        cls_score = []
        bb_pred = []
        kp_pred = []
        for i, layer in enumerate(self.stride_heads):
            score, bb, kp = layer(features[i])
            cls_score.append(score)
            bb_pred.append(bb)
            kp_pred.append(kp)

        return cls_score, bb_pred, kp_pred

    def _forward(self, features: List[Tensor]):
        cls_score = []
        bb_pred = []
        for i, layer in enumerate(self.stride_heads):
            score, bb = layer(features[i])
            cls_score.append(score)
            bb_pred.append(bb)

        return cls_score, bb_pred

    def forward(self, features: List[Tensor]):
        if self.use_kps:
            return self._forward_kps(features)
        else:
            return self._forward(features)
