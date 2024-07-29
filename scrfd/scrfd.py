import torch
import torch.nn as nn
import torchvision
from torch import Tensor
import sys
sys.path.insert(0, '/app/3rd/scrfd_pytorch/scrfd')
from .utils import distance2bbox, distance2kps, parameter_list
from .init_utils import initialize_module_weights
from loguru import logger
from collections import OrderedDict


class SCRFD(nn.Module):
    def __init__(self, backbone: nn.Module, neck: nn.Module, bbox_head: nn.Module):
        super(SCRFD, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.bbox_head = bbox_head

        self.strides = self.bbox_head.strides
        self.num_classes = self.bbox_head.num_classes
        self.use_kps = self.bbox_head.use_kps

        initialize_module_weights(self.neck, validate=True)
        initialize_module_weights(self.bbox_head, validate=True)

    def forward(self, x: Tensor):
        out = self._forward(x)
        return out

    def _forward(self, x: Tensor):
        backbone_feats = self.backbone(x)
        neck_feat = self.neck(backbone_feats)
        outputs = self.bbox_head(neck_feat)
        return outputs

    def preprocess(self, x):
        return x

    @staticmethod
    def get_anchor_centers(fm_h, fm_w, stride, na):
        sy, sx = torch.meshgrid(torch.arange(fm_h), torch.arange(fm_w), indexing='ij')
        anchor_centers = torch.stack([sx, sy], dim=-1).float()
        anchor_centers = (anchor_centers * stride).reshape((-1, 2))
        anchor_centers = torch.stack([anchor_centers] * na, dim=1).reshape((-1, 2))
        return anchor_centers

    def _process_bboxes(self, bbox, stride, anchor_centers):
        b, AxC, h, w = bbox.shape
        na = AxC // 4
        c = 4
        bbox = bbox.view(b, na, c, h, w)
        bbox = bbox.permute(0, 3, 4, 1, 2)
        bbox = bbox.reshape(-1, c)
        bbox *= stride
        bbox = distance2bbox(anchor_centers, bbox)
        return bbox

    def _process_keypoints(self, kps, stride, anchor_centers):
        b, AxC, h, w = kps.shape
        na = AxC // 10
        c = 10
        kps = kps.view(b, na, c, h, w)
        kps = kps.permute(0, 3, 4, 1, 2)
        kps = kps.reshape(-1, c)
        kps *= stride
        kps = distance2kps(anchor_centers, kps)
        return kps

    def _process_scores(self, scores):
        b, AxC, h, w = scores.shape
        na = AxC // self.num_classes
        c = self.num_classes
        scores = scores.view(b, na, c, h, w)
        scores = scores.permute(0, 3, 4, 1, 2)
        scores = scores.reshape(-1, c)
        scores = scores.sigmoid()
        scores, labels = torch.max(scores, dim=1)
        return scores, labels

    def _postprocess_with_keypoints(self, raw_pred, iou_thresh, conf_thresh):
        cls_scores, bboxes, key_points = raw_pred

        per_lvl_preds = []

        bs = cls_scores[0].size()[0]
        for stride_idx, stride in enumerate(self.strides):
            scores = cls_scores[stride_idx]
            bbox = bboxes[stride_idx]
            kps = key_points[stride_idx]

            device = bbox.device
            b, AxC, h, w = bbox.shape
            na = AxC // 4

            anchor_centers = self.get_anchor_centers(h, w, stride, na)
            anchor_centers = anchor_centers.repeat(b, 1).to(device)

            scores, labels = self._process_scores(scores)
            bbox = self._process_bboxes(bbox, stride, anchor_centers)
            kps = self._process_keypoints(kps, stride, anchor_centers)

            batch_ind = torch.arange(b, device=device).view(-1, 1).repeat(1, na * w * h).flatten()  # b x (na*w*h)

            n = bbox.size()[0]
            predictions = torch.zeros((n, 1 + 1 + 1 + 4 + 10), device=device)
            predictions[:, 0] = batch_ind
            predictions[:, 1] = labels
            predictions[:, 2] = scores
            predictions[:, 3:7] = bbox
            predictions[:, 7:] = kps

            per_lvl_preds.append(predictions)

        # confidence filtering
        predictions = torch.cat(per_lvl_preds)
        is_pos = predictions[:, 2] > conf_thresh
        predictions = predictions[is_pos]

        # apply nms
        keep_after_nms = torchvision.ops.batched_nms(
            boxes=predictions[:, 3:7],
            scores=predictions[:, 2],
            idxs=predictions[:, 0],
            iou_threshold=iou_thresh
        )
        predictions = predictions[keep_after_nms]

        out_bboxes = []
        out_scores = []
        out_labels = []
        out_key_points = []
        for i in range(bs):
            pred = predictions[predictions[:, 0] == i]
            out_labels.append(pred[:, 1])
            out_scores.append(pred[:, 2])
            out_bboxes.append(pred[:, 3:7])
            out_key_points.append(pred[:, 7:])

        return out_scores, out_labels, out_bboxes, out_key_points

    def _postprocess_default(self, raw_pred, iou_thresh, conf_thresh):
        cls_scores, bboxes = raw_pred

        per_lvl_preds = []

        bs = cls_scores[0].size()[0]
        for stride_idx, stride in enumerate(self.strides):
            scores = cls_scores[stride_idx]
            bbox = bboxes[stride_idx]

            device = bbox.device
            b, AxC, h, w = bbox.shape
            na = AxC // 4

            anchor_centers = self.get_anchor_centers(h, w, stride, na)
            anchor_centers = anchor_centers.repeat(b, 1).to(device)

            scores, labels = self._process_scores(scores)
            bbox = self._process_bboxes(bbox, stride, anchor_centers)
            batch_ind = torch.arange(b, device=device).view(-1, 1).repeat(1, na * w * h).flatten()  # b x (na*w*h)

            n = bbox.size()[0]
            predictions = torch.zeros((n, 1+1+1+4), device=device)
            predictions[:, 0] = batch_ind
            predictions[:, 1] = labels
            predictions[:, 2] = scores
            predictions[:, 3:7] = bbox

            per_lvl_preds.append(predictions)

        # confidence filtering
        predictions = torch.cat(per_lvl_preds)
        is_pos = predictions[:, 2] > conf_thresh
        predictions = predictions[is_pos]

        # apply nms
        keep_after_nms = torchvision.ops.batched_nms(
            boxes=predictions[:, 3:7],
            scores=predictions[:, 2],
            idxs=predictions[:, 0],
            iou_threshold=iou_thresh
        )
        predictions = predictions[keep_after_nms]

        out_bboxes = []
        out_scores = []
        out_labels = []
        for i in range(bs):
            pred = predictions[predictions[:, 0] == i]
            out_labels.append(pred[:, 1])
            out_scores.append(pred[:, 2])
            out_bboxes.append(pred[:, 3:7])

        return out_scores, out_labels, out_bboxes

    def postprocess(self, raw_pred, iou_thresh, conf_thresh):
        if self.use_kps:
            return self._postprocess_with_keypoints(raw_pred, iou_thresh, conf_thresh)
        else:
            return self._postprocess_default(raw_pred, iou_thresh, conf_thresh)

    def load_from_checkpoint(self, ckpt_fp, strict=True, verbose=True, partial=False):
        ckpt = torch.load(ckpt_fp)

        if partial:
            if verbose:
                logger.warning('Partial checkpoint initialization: backbone and neck are loaded only')
            b_sd = OrderedDict()
            # -------- Backbone
            for k, v in ckpt['model'].items():
                if k.startswith('backbone'):
                    b_sd[k[9:]] = v
            self.backbone.load_state_dict(b_sd)

            # -------- NECK
            # n_sd = OrderedDict()
            # for k, v in ckpt['model'].items():
            #     if k.startswith('neck'):
            #         n_sd[k[5:]] = v
            # self.neck.load_state_dict(n_sd)
        else:
            self.load_state_dict(state_dict=ckpt['model'], strict=strict)
        del ckpt

    def get_param_groups(self, no_decay_bn_filter_bias, wd):
        return parameter_list(self.named_parameters, weight_decay=wd, no_decay_bn_filter_bias=no_decay_bn_filter_bias)
