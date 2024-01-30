import math
import torch
from .utils import bbox_overlaps


class ATSSModule(object):
    def __init__(self, top_k):
        self.top_k = top_k

    def __call__(self, anchors_per_im, bboxes_per_im, num_anchors_per_level):

        num_gt = len(bboxes_per_im)
        ious = bbox_overlaps(anchors_per_im, bboxes_per_im, mode='iou')

        # compute gt centers
        gt_cx = (bboxes_per_im[:, 2] + bboxes_per_im[:, 0]) / 2.0
        gt_cy = (bboxes_per_im[:, 3] + bboxes_per_im[:, 1]) / 2.0
        gt_points = torch.stack((gt_cx, gt_cy), dim=1)

        # compute gt areas
        gt_width = bboxes_per_im[:, 2] - bboxes_per_im[:, 0]
        gt_height = bboxes_per_im[:, 3] - bboxes_per_im[:, 1]
        gt_area = torch.sqrt(torch.clamp(gt_width * gt_height, min=1e-4))

        # compute anchor centers
        anchors_cx_per_im = (anchors_per_im[:, 2] + anchors_per_im[:, 0]) / 2.0
        anchors_cy_per_im = (anchors_per_im[:, 3] + anchors_per_im[:, 1]) / 2.0
        anchor_points = torch.stack((anchors_cx_per_im, anchors_cy_per_im), dim=1)

        # compute center distance between anchor box and object
        distances = (anchor_points[:, None, :] - gt_points[None, :, :]).pow(2).sum(-1).sqrt()

        # selecting candidates based on the center distance between anchor box and object
        candidate_idxs = []
        star_idx = 0
        for na in num_anchors_per_level:
            end_idx = star_idx + na
            distances_per_level = distances[star_idx:end_idx, :]

            selectable_k = min(self.top_k, na)

            _, topk_idxs_per_level = distances_per_level.topk(selectable_k, dim=0, largest=False)
            candidate_idxs.append(topk_idxs_per_level + star_idx)
            star_idx = end_idx

        candidate_idxs = torch.cat(candidate_idxs, dim=0)

        # Using the sum of mean and standard deviation as the IoU threshold to select final positive samples
        candidate_ious = ious[candidate_idxs, torch.arange(num_gt)]
        iou_mean_per_gt = candidate_ious.mean(0)
        iou_std_per_gt = candidate_ious.std(0)
        iou_thresh_per_gt = iou_mean_per_gt + iou_std_per_gt
        is_pos = candidate_ious >= iou_thresh_per_gt[None, :]

        # Limiting the final positive samples’ center to object
        anchor_num = anchors_cx_per_im.shape[0]
        for ng in range(num_gt):
            candidate_idxs[:, ng] += ng * anchor_num
        e_anchors_cx = anchors_cx_per_im.view(1, -1).expand(num_gt, anchor_num).contiguous().view(-1)
        e_anchors_cy = anchors_cy_per_im.view(1, -1).expand(num_gt, anchor_num).contiguous().view(-1)
        candidate_idxs = candidate_idxs.view(-1)
        l_ = e_anchors_cx[candidate_idxs].view(-1, num_gt) - bboxes_per_im[:, 0]
        t_ = e_anchors_cy[candidate_idxs].view(-1, num_gt) - bboxes_per_im[:, 1]
        r_ = bboxes_per_im[:, 2] - e_anchors_cx[candidate_idxs].view(-1, num_gt)
        b_ = bboxes_per_im[:, 3] - e_anchors_cy[candidate_idxs].view(-1, num_gt)

        dist_min = torch.stack([l_, t_, r_, b_], dim=1).min(dim=1)[0]
        dist_min = dist_min.div_(gt_area)

        is_in_gts = dist_min > 0.001
        is_pos = is_pos & is_in_gts

        # if an anchor box is assigned to multiple gts, the one with the highest IoU will be selected.
        ious_inf = torch.full_like(ious, -math.inf).t().contiguous().view(-1)
        index = candidate_idxs.view(-1)[is_pos.view(-1)]
        ious_inf[index] = ious.t().contiguous().view(-1)[index]
        ious_inf = ious_inf.view(num_gt, -1).t()

        anchors_to_gt_values, anchors_to_gt_indexs = ious_inf.max(dim=1)

        return anchors_to_gt_indexs, anchors_to_gt_values
