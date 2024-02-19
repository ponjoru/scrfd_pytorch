import torch
import collections
import torch.nn as nn
import torchvision

from torch import Tensor
from itertools import repeat
from typing import Optional


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse

_pair = _ntuple(2)


class ConvBnAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, dilation=1, groups=1, with_norm=True, activation=nn.ReLU):
        super(ConvBnAct, self).__init__()

        stride = _pair(stride)
        dilation = _pair(dilation)
        kernel_size = _pair(kernel_size)
        conv_bias = with_norm is False

        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation, groups, bias=conv_bias)
        self.bn = nn.Identity() if not with_norm else nn.BatchNorm2d(out_ch)
        self.act = activation()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class DWConvBnAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, dilation=1, with_norm=True, activation=nn.ReLU):
        super(DWConvBnAct, self).__init__()

        self.depthwise_conv = ConvBnAct(
            in_ch,
            in_ch,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_ch,
            with_norm=with_norm,
            activation=activation,
        )

        self.pointwise_conv = ConvBnAct(
            in_ch,
            out_ch,
            1,
            with_norm=with_norm,
            activation=activation
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
    """Calculate overlap between two set of bboxes.

    If ``is_aligned `` is ``False``, then calculate the overlaps between each
    bbox of bboxes1 and bboxes2, otherwise the overlaps between each aligned
    pair of bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
            B indicates the batch dim, in shape (B1, B2, ..., Bn).
            If ``is_aligned `` is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union), "iof" (intersection over
            foreground) or "giou" (generalized intersection over union).
            Default "iou".
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.
        eps (float, optional): A value added to the denominator for numerical
            stability. Default 1e-6.

    Returns:
        Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)

    """

    assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes's last dimenstion is 4
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows, ))
        else:
            return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])

    if is_aligned:
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

        wh = (rb - lt).clamp(min=0)  # [B, rows, 2]
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = torch.max(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = torch.min(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

        wh = (rb - lt).clamp(min=0)  # [B, rows, cols, 2]
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])
            enclosed_rb = torch.max(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ['iou', 'iof']:
        return ious
    # calculate gious
    enclose_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return torch.stack([x1, y1, x2, y2], dim=-1)


def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i%2] + distance[:, i]
        py = points[:, i%2+1] + distance[:, i+1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return torch.stack(preds, dim=-1)


def kps2distance(points, kps, max_dis=None, eps=0.1):
    """Decode bounding box based on distances.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        kps (Tensor): Shape (n, K), "xyxy" format
        max_dis (float): Upper bound of the distance.
        eps (float): a small value to ensure target < max_dis, instead <=

    Returns:
        Tensor: Decoded distances.
    """

    preds = []
    for i in range(0, kps.shape[1], 2):
        px = kps[:, i] - points[:, i%2]
        py = kps[:, i+1] - points[:, i%2+1]
        if max_dis is not None:
            px = px.clamp(min=0, max=max_dis - eps)
            py = py.clamp(min=0, max=max_dis - eps)
        preds.append(px)
        preds.append(py)
    return torch.stack(preds, -1)


def parameter_list(
    named_parameters,
    weight_decay: Optional[float] = 0.0,
    no_decay_bn_filter_bias: Optional[bool] = False,
    *args,
    **kwargs
):
    with_decay = []
    without_decay = []
    if isinstance(named_parameters, list):
        for n_parameter in named_parameters:
            for p_name, param in n_parameter():
                if (
                    param.requires_grad
                    and len(param.shape) == 1
                    and no_decay_bn_filter_bias
                ):
                    # biases and normalization layer parameters are of len 1
                    without_decay.append(param)
                elif param.requires_grad:
                    with_decay.append(param)
    else:
        for p_name, param in named_parameters():
            if (
                param.requires_grad
                and len(param.shape) == 1
                and no_decay_bn_filter_bias
            ):
                # biases and normalization layer parameters are of len 1
                without_decay.append(param)
            elif param.requires_grad:
                with_decay.append(param)
    param_list = [{"params": with_decay, "weight_decay": weight_decay}]
    if len(without_decay) > 0:
        param_list.append({"params": without_decay, "weight_decay": 0.0})
    return param_list


def get_anchor_centers(fm_h, fm_w, stride, na):
    sy, sx = torch.meshgrid(torch.arange(fm_w), torch.arange(fm_h), indexing='ij')
    anchor_centers = torch.stack([sx, sy], dim=-1).float()
    anchor_centers = (anchor_centers * stride).reshape((-1, 2))
    anchor_centers = torch.stack([anchor_centers] * na, dim=1).reshape((-1, 2))
    return anchor_centers


def postprocess(raw_pred, iou_thresh, conf_thresh):
    num_classes = 1
    strides = (8, 16, 32)
    cls_scores, bboxes, key_points, _ = raw_pred

    bb_per_lvl = []
    kp_per_lvl = []
    scores_per_lvl = []
    labels_per_lvl = []
    per_lvl_batch_ind = []

    for stride_idx, stride in enumerate(strides):
        bbox = bboxes[stride_idx]
        kps = key_points[stride_idx]

        device = bbox.device

        b, AxC, h, w = bbox.shape
        na = AxC // 4
        c = 4
        bbox = bbox.view(b, na, c, h, w)
        bbox = bbox.permute(0, 3, 4, 1, 2)
        bbox = bbox.reshape(-1, c)
        bbox *= stride

        anchor_centers = get_anchor_centers(h, w, stride, na)
        anchor_centers = anchor_centers.repeat(b, 1)
        anchor_centers = anchor_centers.to(device)

        bbox = distance2bbox(anchor_centers, bbox)
        bb_per_lvl.append(bbox)

        b, AxC, h, w = kps.shape
        na = AxC // 10
        c = 10
        kps = kps.view(b, na, c, h, w)
        kps = kps.permute(0, 3, 4, 1, 2)
        kps = kps.reshape(-1, c)
        kps *= stride

        kps = distance2kps(anchor_centers, kps)
        kp_per_lvl.append(kps)

        scores = cls_scores[stride_idx]
        b, AxC, h, w = scores.shape
        na = AxC // num_classes
        c = num_classes
        scores = scores.view(b, na, c, h, w)
        scores = scores.permute(0, 3, 4, 1, 2)
        scores = scores.reshape(-1, c)
        scores = scores.sigmoid()
        scores, labels = torch.max(scores, dim=1)

        scores_per_lvl.append(scores)
        labels_per_lvl.append(labels)

        batch_ind = torch.arange(b, device=device).view(-1, 1).repeat(1, na * w * h).flatten()  # b x (na*w*h)
        per_lvl_batch_ind.append(batch_ind)

    bboxes = torch.cat(bb_per_lvl)
    kps = torch.cat(kp_per_lvl)
    scores = torch.cat(scores_per_lvl)
    labels = torch.cat(labels_per_lvl)
    idxs = torch.cat(per_lvl_batch_ind)

    # todo: returns tuple check?
    is_pos = torch.where(scores > conf_thresh)[0]
    bboxes = bboxes[is_pos]
    kps = kps[is_pos]
    scores = scores[is_pos]
    labels = labels[is_pos]
    idxs = idxs[is_pos]

    keep_after_nms = torchvision.ops.batched_nms(bboxes, scores, idxs, iou_threshold=iou_thresh)

    bboxes = bboxes[keep_after_nms]
    kps = kps[keep_after_nms]
    scores = scores[keep_after_nms]
    labels = labels[keep_after_nms]
    idxs = idxs[keep_after_nms]

    out_bboxes = [[] for _ in range(b)]
    out_scores = [[] for _ in range(b)]
    out_labels = [[] for _ in range(b)]
    out_key_points = [[] for _ in range(b)]
    for bbox, kp, score, lbl, idx in zip(bboxes, kps, scores, labels, idxs):
        out_scores[idx].append(score)
        out_bboxes[idx].append(bbox)
        out_key_points[idx].append(kp)
        out_labels[idx].append(lbl)
    # cls_score, labels, bbox_pred, kps
    return out_scores, out_labels, out_bboxes, out_key_points
