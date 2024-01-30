import math
import random
import numpy as np

from typing import Dict, Tuple, Any
from albumentations import DualTransform
from albumentations.augmentations.crops import functional as F
from albumentations.core.transforms_interface import BoxInternalType
from albumentations.core.bbox_utils import denormalize_bboxes


def matrix_iof(a, b):
    """
    return iof of a and b, numpy version for data augmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    return area_i / np.maximum(area_a[:, np.newaxis], 1)


class BBSafeScaledRandomCrop(DualTransform):
    """
    Modified bbox safe scaled random crop (re-implementation from SCRFD paper). Performs 250 attempts to generate
    random scaled crop with at least `min_boxes_kept` bbox in it. Otherwise returns non-processed image
    """
    # scrfd default scales: [0.3, 0.45, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    # tinaface default scales: [0.3, 0.45, 0.6, 0.8, 1.0]
    _DEFAULT_SCALES = [0.3, 0.45, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

    def __init__(self, scales=None, ratio=(0.75, 1.3333333333333333), min_boxes_kept=1, always_apply=False, p=0.5):
        super(BBSafeScaledRandomCrop, self).__init__(always_apply=always_apply, p=p)
        self.scales = scales if scales is not None else self._DEFAULT_SCALES
        self.ratio = ratio
        self.min_boxes_kept = min_boxes_kept

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return 'scales', 'ratio', 'min_boxes_kept'

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        img_h, img_w = params["image"].shape[:2]
        boxes = params["bboxes"]
        boxes = denormalize_bboxes(boxes, rows=img_w, cols=img_h)

        area = img_w * img_h
        for _ in range(250):
            scale = random.choice(self.scales)
            target_area = scale * area
            log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))  # skipcq: PTC-W0028
            h = int(round(math.sqrt(target_area / aspect_ratio)))  # skipcq: PTC-W0028

            if 0 < w <= img_w and 0 < h <= img_h:
                i = random.randint(0, img_h - h)
                j = random.randint(0, img_w - w)

                w_start = j * 1.0 / (img_w - w + 1e-10)
                h_start = i * 1.0 / (img_h - h + 1e-10)

                if len(boxes) == 0:
                    return {'crop_height': h, 'crop_width': w, 'h_start': h_start, 'w_start': w_start}

                boxes = np.stack([b[:4] for b in boxes])
                x1, y1 = w_start * img_w, h_start * img_h
                roi = np.array((x1, y1, x1 + w, y1 + h))
                iof = matrix_iof(boxes, roi[np.newaxis])

                num_boxes_inside_roi = sum(iof == 1.0)
                if num_boxes_inside_roi >= self.min_boxes_kept:
                    return {'crop_height': h, 'crop_width': w, 'h_start': h_start, 'w_start': w_start}

        # fallback to unmodified image
        return {'crop_height': img_h, 'crop_width': img_w, 'h_start': 0, 'w_start': 0}

    def apply(self, img, crop_height=0, crop_width=0, h_start=0, w_start=0, **params):
        crop = F.random_crop(img, crop_height, crop_width, h_start, w_start)
        return crop

    def apply_to_bbox(self, bbox, crop_height=0, crop_width=0, h_start=0, w_start=0, rows=0, cols=0, **params):
        return F.bbox_random_crop(bbox, crop_height, crop_width, h_start, w_start, rows, cols)

    def apply_to_keypoint(self, keypoint, crop_height=0, crop_width=0, h_start=0, w_start=0, rows=0, cols=0, **params):
        keypoint = F.keypoint_random_crop(keypoint, crop_height, crop_width, h_start, w_start, rows, cols)
        return keypoint

    @property
    def targets_as_params(self):
        return ["image", "bboxes"]
