import math
import torch
import torch.nn as nn
import numpy as np
from torch import Tensor

from typing import List, Tuple, Dict, Optional


from .utils import bbox_overlaps, distance2bbox, kps2distance


def reshape_predictions(pred, num_channels):
    N, AxC, H, W = pred.shape
    A = AxC // num_channels
    C = num_channels
    pred = pred.view(N, A, C, H, W)
    pred = pred.permute(0, 3, 4, 1, 2)
    pred = pred.reshape(N, -1, C)
    return pred


class SCRFDLoss(nn.Module):
    """
    SCRFD loss re-implementation.

    bbox format:
        predictions: [xyxy]
        targets: [xyxyw] - x1, y1, x2, y2, bbox_weight
    keypoints format:
        predictions: [xy]
        targets: [xyw] - x, y, keypoint_weight

    Args:
        cls_loss (nn.Module): classification loss
        bb_loss (nn.Module): bounding box loss
        anchor_generator (object) : anchor generator
        matcher (object): targets and anchors matcher
        kp_loss (Optional[nn.Module]): keypoints loss (default = None)
        num_classes (int): number of classes (default = 1)
        strides (List[int]): tuple of strides to process (default = (8, 16, 32))
        use_kps (bool): whether to use keypoints branch (default = True)
        use_qscore: whether to use quality score (default = True), predict IoU between pred and target to
                    refine the confidence score
    """
    _BATCH_ID = 0
    _LABEL_ID = 1
    _BBOX_WEIGHT = 2
    _START_BBOX = 3
    _START_KPS = 7
    _NK = 5

    def __init__(
            self,
            cls_loss: nn.Module,
            bb_loss: nn.Module,
            anchor_generator: object,
            matcher: object,
            kp_loss: Optional[nn.Module] = None,
            num_classes: int = 1,
            strides: List[int] = (8, 16, 32),
            use_kps: bool = True,
            use_qscore: bool = True
    ):
        super(SCRFDLoss, self).__init__()
        self.cls_loss = cls_loss
        self.bb_reg_loss = bb_loss
        self.kp_reg_loss = kp_loss
        self.matcher = matcher
        self.anchor_generator = anchor_generator

        self.loss_kps_std = 1.0

        self.strides = strides
        self.num_classes = num_classes

        self.use_kps = use_kps
        self.use_qscore = use_qscore

    def _prepare_predictions(self, predictions: List[Tensor]) -> List[Tensor]:
        """
        Preprocess predictions to per stride format

        Args:
            predictions (List[Tensor(N,C_i,H,W)]): cls scores - [N,C1,H,W], bbox predictions - [N,C2,H,W] and (optional)
                                                   keypoint predictions - [N,C3,H,W],
                                                   where C1 = num_anchors * num_classes, C2 = num_anchors * 4,
                                                   C3 = num_anchors * 2 * num_keypoints
        Returns:
            per_stride_predictions List[Tensor(N, num_candidates, C)]: List of per stride predictions, where
            C = len(cls_scores, bbox[xyxy], keypoints[xy])
        """
        if self.use_kps:
            cls_pred, bb_pred, kp_pred = predictions
        else:
            cls_pred, bb_pred = predictions

        items = []

        per_stride_cls = [reshape_predictions(p, num_channels=self.num_classes) for p in cls_pred]
        items.append(per_stride_cls)
        per_stride_bb = [reshape_predictions(p, num_channels=4) for p in bb_pred]
        items.append(per_stride_bb)

        if self.use_kps:
            per_stride_kp = [reshape_predictions(p, num_channels=self._NK * 2) for p in kp_pred]
            items.append(per_stride_kp)

        n = len(self.strides)
        m = len(items)

        per_stride_predictions = []
        for i in range(n):
            per_stride_pred = torch.cat([items[j][i] for j in range(m)], dim=-1)
            per_stride_predictions.append(per_stride_pred)
        return per_stride_predictions

    def _prepare_targets(self, targets: Tensor, anchors: List[Tensor], batch_size: int) -> List[Tensor]:
        """
        Preprocess targets to per stride format and assign them to anchors (tgt-anchor: one-to-many) with a matcher
        Args:
            targets (Tensor[N,C]):  bbox targets (w/ keypoints optional),
                                    where C = len(img_id, label, bbox[xyxyw], keypoints[xyw]).
            anchors (List[Tensor(NA_i, 4)]): list of per stride anchors, NA_i = num_anchors per stride i
        Returns:
            per_stride_targets (List[Tensor(N, NA_i, C)]): per stride targets
        """
        num_anchors_per_level = [len(_) for _ in anchors]
        anchors_per_im = torch.cat(anchors).to(targets.device)

        targets_list = []
        for im_i in range(batch_size):
            target_per_im = targets[targets[:, self._BATCH_ID] == im_i]

            if len(target_per_im) == 0:
                n, d = target_per_im.size()
                target_per_im = torch.zeros((len(anchors_per_im), d)).to(targets.device)
                target_per_im[:, self._LABEL_ID] = self.num_classes
                targets_list.append(target_per_im)
                continue

            # match indices
            start_idx, end_idx = self._START_BBOX, self._START_BBOX + 4
            bb_per_im = target_per_im[:, start_idx:end_idx]
            anchors_to_gt_indices, anchors_to_gt_values = self.matcher(anchors_per_im, bb_per_im, num_anchors_per_level)

            # select & add class labels
            target_per_im = target_per_im[anchors_to_gt_indices]
            target_per_im[:, self._LABEL_ID][anchors_to_gt_values == -math.inf] = self.num_classes

            targets_list.append(target_per_im)

        # split targets and weights by levels
        start_idx = 0
        per_stride_targets = []
        for na in num_anchors_per_level:
            end_idx = start_idx + na
            per_stride_target = torch.stack(targets_list)[:, start_idx:end_idx, :]
            per_stride_targets.append(per_stride_target)
            start_idx = end_idx

        return per_stride_targets

    @staticmethod
    def anchor_center(anchors):
        """Get anchor centers from anchors.

        Args:
            anchors (Tensor): Anchor list with shape (N, 4), "xyxy" format.

        Returns:
            Tensor: Anchor centers with shape (N, 2), "xy" format.
        """
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        return torch.stack([anchors_cx, anchors_cy], dim=-1)

    def _get_num_total_pos_targets(self, targets: List[Tensor]) -> int:
        """ Get total number of positive targets across all strides

        Args:
             targets (List[Tensor(N, NA_i, C)]): per stride targets
        Returns:
             n (int): number of positive targets
        """
        n = 0
        for stride_t in targets:
            labels = stride_t[:, :, self._LABEL_ID]
            n += int(((labels >= 0) & (labels < self.num_classes)).sum())
        return n

    def _compute_cls_loss(self, predictions: Tensor, targets: Tensor, cls_label_scores: Tensor, num_total_samples: int) -> Dict[str, Tensor]:
        labels = targets[..., self._LABEL_ID]
        weights = targets[..., self._BBOX_WEIGHT]
        cls_score = predictions[..., :self.num_classes].reshape(-1, self.num_classes)

        num_total_samples = max(num_total_samples, 1.0)
        loss_cls = self.cls_loss(
            cls_score, (labels, cls_label_scores),
            weight=weights,
            avg_factor=num_total_samples
        )

        loss_name = f'cls_{self.cls_loss.name}'
        loss_dict = {
            loss_name: loss_cls
        }
        return loss_dict

    def _compute_bb_loss(self, predictions: Tensor, targets: Tensor, anchors: Tensor, stride: int) -> Dict[str, Tensor]:
        # predictions:  [n, nc + 4 + nk*2]
        # targets:      [n, id + 1 + (4+1) + nk*3]
        anchor_centers = self.anchor_center(anchors) / stride

        nc = self.num_classes
        gt_bbox = targets[..., self._START_BBOX:self._START_BBOX+4]
        dt_bbox = predictions[..., nc:nc+4]
        dt_scores = predictions[..., :nc]
        dt_scores = dt_scores.detach().sigmoid()
        dt_scores, _ = dt_scores.max(dim=1)

        weights = targets[..., self._BBOX_WEIGHT] * dt_scores

        decoded_gt_bbox = gt_bbox / stride
        decoded_dt_bbox = distance2bbox(anchor_centers, dt_bbox)
        loss_bbox = self.bb_reg_loss(decoded_dt_bbox, decoded_gt_bbox, weight=weights, avg_factor=1.0)

        loss_name = f'bb_{self.bb_reg_loss.name}'
        loss_dict = {
            loss_name: loss_bbox
        }
        return loss_dict, dt_scores

    def _compute_kp_loss(self, predictions: Tensor, targets: Tensor, anchors: Tensor, stride: int) -> Dict[str, Tensor]:
        # predictions:  [n, nc + 4 + nk*2]
        # targets:      [n, id + 1 + (4+1) + nk*3]
        anchor_centers = self.anchor_center(anchors) / stride

        nc = self.num_classes
        dt_scores = predictions[..., :nc]
        dt_scores = dt_scores.detach().sigmoid()
        dt_scores, _ = dt_scores.max(dim=1)

        kps = targets[..., self._START_KPS:]
        kps = kps.reshape(-1, self._NK, 3)

        gt_kps = kps[..., :2].reshape(-1, self._NK * 2)
        dt_kps = predictions[..., nc+4:].reshape(-1, self._NK * 2)

        weights = kps[..., 2] * dt_scores.unsqueeze(1)

        decoded_gt_kps = kps2distance(anchor_centers, gt_kps / stride) * self.loss_kps_std
        decoded_dt_kps = dt_kps * self.loss_kps_std

        # after
        decoded_gt_kps = decoded_gt_kps.reshape(-1, 2)
        decoded_dt_kps = decoded_dt_kps.reshape(-1, 2)
        weights = weights.reshape(-1, 1)
        loss_kps = self.kp_reg_loss(decoded_dt_kps, decoded_gt_kps, weight=weights, avg_factor=1.0)

        loss_name = f'kp_{self.kp_reg_loss.name}'
        loss_dict = {
            loss_name: loss_kps
        }
        return loss_dict

    def _compute_qs_score(self, predictions, targets, anchors, stride):
        anchor_centers = self.anchor_center(anchors) / stride

        nc = self.num_classes
        gt_bbox = targets[..., self._START_BBOX:self._START_BBOX+4]
        dt_bbox = predictions[..., nc:nc + 4]

        decoded_gt_bbox = gt_bbox / stride
        decoded_dt_bbox = distance2bbox(anchor_centers, dt_bbox)

        score = bbox_overlaps(decoded_dt_bbox.detach(), decoded_gt_bbox, is_aligned=True)
        return score

    def compute_losses(self, predictions, targets, anchors, stride, num_total_samples):
        # targets:       [N, id+l+(4+1)+(3*5)]      (N, 22) or (N, 7)
        # predictions:   [N, id+sc+4+2*5]       # (N2, 15) or (N2, 5)
        device = predictions.device

        targets = targets.reshape(-1, targets.size()[-1])
        predictions = predictions.reshape(-1, predictions.size()[-1])
        anchors = anchors.reshape(-1, anchors.size()[-1])

        labels = targets[..., 1]
        cls_label_score = targets.new_zeros(labels.shape, dtype=torch.float32)

        pos_inds = ((labels >= 0) & (labels < self.num_classes)).nonzero().squeeze(1)

        if len(pos_inds) > 0:
            pos_anchors = anchors[pos_inds]
            pos_targets = targets[pos_inds]
            pos_predictions = predictions[pos_inds]

            loss_dict, weight_targets = self._compute_bb_loss(pos_predictions, pos_targets, pos_anchors, stride)

            if self.use_kps:
                kp_loss_dict = self._compute_kp_loss(pos_predictions, pos_targets, pos_anchors, stride)
                loss_dict = {**loss_dict, **kp_loss_dict}

            if self.use_qscore:
                cls_label_score[pos_inds] = self._compute_qs_score(pos_predictions, pos_targets, pos_anchors, stride)
            else:
                cls_label_score[pos_inds] = 1.0
        else:
            bb_loss_name = f'bb_{self.bb_reg_loss.name}'
            loss_dict = {bb_loss_name: torch.tensor(0).to(device)}
            if self.use_kps:
                kp_loss_name = f'kp_{self.kp_reg_loss.name}'
                loss_dict[kp_loss_name] = torch.tensor(0).to(device)

            weight_targets = torch.tensor(0).to(device)

        cls_loss_dict = self._compute_cls_loss(predictions, targets, cls_label_score, num_total_samples)

        loss_dict = {**cls_loss_dict, **loss_dict}

        return loss_dict, weight_targets.sum()

    @staticmethod
    def aggregate_losses(losses: List[Dict[str, Tensor]], avg_factor: float) -> Dict[str, Tensor]:
        """
        Aggregate losses across the strides (sum(losses) / avg_factor). Check original SCRFD implementation for
        the details

        Args:
            losses (List[Dict[str, Tensor]]): dict of losses across all strides
            avg_factor: if avg_factor == 1.0 just sum aggregation is applied
        Returns:
            loss_dict (Dict[str, Tensor]): aggregated losses
        """
        names = losses[0].keys()
        n = len(losses)
        m = len(names)

        losses = torch.stack([l for loss_dict in losses for l in loss_dict.values()])
        losses = losses.reshape(n, m)
        losses[:, 1:] /= avg_factor  # check out SCRFD implementation

        loss_items = losses.mean(dim=0)
        loss_dict = {}
        for ind, loss_name in enumerate(names):
            loss_dict[loss_name] = loss_items[ind]
        return loss_dict

    def forward(self, predictions: List[Tensor], targets: Tensor, meta: dict) -> Dict[str, Tensor]:
        """

        Args:
            predictions (List[Tensor(N,C_i,H,W)]): SCRFD model predictions (cls_score, bb_pred, kp_pred (optional))
            targets(Tensor[N,C]):  bbox targets (w/ keypoints optional),
                                   single target format: len(img_id, label, bbox[xyxyw], keypoints[xyw]).
            meta (dict): dict with batch meta information, must include current batch shape key ('img1_shape')
        Returns:
            loss_dict (Dict[str, Tensor]): averaged across the strides losses, key - loss name, value - loss value
        """
        bs = len(meta['img_path'])
        img_size = meta['img1_shape']

        # prepare input predictions and targets, assign targets to anchors
        predictions = self._prepare_predictions(predictions)
        # generate anchors
        anchors = self.anchor_generator(img_size=img_size, device=targets.device)
        # prepare targets and assign them to anchors
        targets = self._prepare_targets(targets, anchors, batch_size=bs)

        num_total_samples = self._get_num_total_pos_targets(targets)

        loss_list = []
        wts = []
        for i, stride in enumerate(self.strides):
            stride_dt = predictions[i]
            stride_gt = targets[i]
            stride_an = anchors[i].unsqueeze(0).repeat((bs, 1, 1))

            # compute losses per stride
            loss_dict, wt = self.compute_losses(stride_dt, stride_gt, stride_an, stride=stride, num_total_samples=num_total_samples)
            loss_list.append(loss_dict)
            wts.append(wt)

        loss_dict = self.aggregate_losses(loss_list, avg_factor=max(1.0, sum(wts)))
        return loss_dict

    @property
    def num_loss_items(self):
        nl = 3 if self.use_kps else 2
        return nl

