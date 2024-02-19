# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math

import numpy as np
import torch
from torch import nn


class BufferList(nn.Module):
    """
    Similar to nn.ParameterList, but for buffers
    """

    def __init__(self, buffers=None):
        super(BufferList, self).__init__()
        if buffers is not None:
            self.extend(buffers)

    def extend(self, buffers):
        offset = len(self)
        for i, buffer in enumerate(buffers):
            self.register_buffer(str(offset + i), buffer)
        return self

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())


def generate_anchors(base_size, scales, aspect_ratios):
    w = base_size
    h = base_size
    x_center, y_center = 0, 0

    h_ratios = np.sqrt(aspect_ratios)
    w_ratios = 1 / h_ratios

    ws = (w * w_ratios[:, None] * scales[None, :]).flatten()
    hs = (h * h_ratios[:, None] * scales[None, :]).flatten()

    # use float anchor and the anchor's center is aligned with the
    # pixel center
    base_anchors = [
        x_center - 0.5 * ws,
        y_center - 0.5 * hs,
        x_center + 0.5 * ws,
        y_center + 0.5 * hs
    ]
    base_anchors = np.stack(base_anchors, axis=-1)
    return torch.from_numpy(base_anchors)


class AnchorGenerator(object):
    """
    For a set of image sizes and feature maps, computes a set
    of anchors
    """

    def __init__(
        self,
        scales=(1., 2.),
        sizes=(128, 256, 512),
        aspect_ratios=(0.5, 1.0, 2.0),
        anchor_strides=(8, 16, 32),
        straddle_thresh=0,
    ):
        super(AnchorGenerator, self).__init__()

        if len(anchor_strides) == 1:
            anchor_stride = anchor_strides[0]
            cell_anchors = [
                generate_anchors(anchor_stride, sizes, aspect_ratios).float()
            ]
        else:
            if len(anchor_strides) != len(sizes):
                raise RuntimeError("FPN should have #anchor_strides == #sizes")

            cell_anchors = []
            for anchor_stride, size in zip(anchor_strides, sizes):
                a = generate_anchors(
                    scales=np.array(scales, dtype=np.float32),
                    base_size=np.array(size, dtype=np.float32),
                    aspect_ratios=np.array(aspect_ratios, dtype=np.float32)
                ).float()
                cell_anchors.append(a)

        self.strides = anchor_strides
        self.cell_anchors = BufferList(cell_anchors)
        self.straddle_thresh = straddle_thresh

    def num_anchors_per_location(self):
        return [len(cell_anchors) for cell_anchors in self.cell_anchors]

    def grid_anchors(self, grid_sizes, device):
        anchors = []
        for size, stride, base_anchors in zip(grid_sizes, self.strides, self.cell_anchors):
            grid_height, grid_width = size
            shifts_x = torch.arange(0, grid_width * stride, step=stride, dtype=torch.float32, device=device)
            shifts_y = torch.arange(0, grid_height * stride, step=stride, dtype=torch.float32, device=device)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            anchors.append(
                (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4).to(device)).reshape(-1, 4)
            )

        return anchors

    def __call__(self, img_size, device):
        grid_sizes = [(int(img_size[0] / s), int(img_size[1] / s)) for s in self.strides]
        return self.grid_anchors(grid_sizes, device)


def generate_custom_anchors(ws, hs):
    x_center, y_center = 0, 0

    # use float anchor and the anchor's center is aligned with the
    # pixel center
    base_anchors = [
        x_center - 0.5 * ws,
        y_center - 0.5 * hs,
        x_center + 0.5 * ws,
        y_center + 0.5 * hs
    ]
    base_anchors = np.stack(base_anchors, axis=-1)
    return torch.from_numpy(base_anchors)


class CustomAnchorGenerator(object):
    """
    For a set of image sizes and feature maps, computes a set
    of anchors
    """

    def __init__(
        self,
        anchor_sizes,
        anchor_strides=(8, 16, 32),
    ):
        super(CustomAnchorGenerator, self).__init__()

        ns = len(anchor_strides)

        assert len(anchor_sizes) % ns == 0, f'Number of anchor sizes should be divisible by strides number, got {ns} and {len(anchor_sizes)}'
        sizes = anchor_sizes
        sizes = np.array(sizes).reshape(ns, -1, 2)
        cell_anchors = []
        for i, stride in enumerate(anchor_strides):
            ws = sizes[i, :, 0]
            hs = sizes[i, :, 1]
            a = generate_custom_anchors(ws, hs).float()
            cell_anchors.append(a)

        self.strides = anchor_strides
        self.cell_anchors = BufferList(cell_anchors)

    def num_anchors_per_location(self):
        return [len(cell_anchors) for cell_anchors in self.cell_anchors]

    def grid_anchors(self, grid_sizes, device):
        anchors = []
        for size, stride, base_anchors in zip(grid_sizes, self.strides, self.cell_anchors):
            grid_height, grid_width = size
            shifts_x = torch.arange(0, grid_width * stride, step=stride, dtype=torch.float32, device=device)
            shifts_y = torch.arange(0, grid_height * stride, step=stride, dtype=torch.float32, device=device)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            anchors.append(
                (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4).to(device)).reshape(-1, 4)
            )

        return anchors

    def __call__(self, img_size, device):
        grid_sizes = [(int(img_size[0] / s), int(img_size[1] / s)) for s in self.strides]
        return self.grid_anchors(grid_sizes, device)


if __name__ == '__main__':
    a1 = CustomAnchorGenerator(
        anchor_sizes=[14,35, 25,62, 42,106, 67,166, 105,263, 162,424, 282,651, 539,1222, 1127,1764]
    )
    a2 = AnchorGenerator(
        scales=[1., 2.],
        aspect_ratios=[1.0],
        sizes=[16, 64, 256],
        anchor_strides=[8, 16, 32],
    )

    r1 = a1(img_size=(352, 640), device='cpu')
    r2 = a2(img_size=(352, 640), device='cpu')
    k = 5