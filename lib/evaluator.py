"""
WiderFace evaluation code
author: wondervictor
mail: tianhengcheng@gmail.com
copyright@wondervictor
"""

# from __future__ import absolute_import
import os
import tqdm
import pickle
import datetime
import argparse
import torch
import numpy as np
from scipy.io import loadmat
from collections import defaultdict
from multiprocessing import Pool

from typing import List, Dict, Tuple
from loguru import logger


from lib.utils import DSMetrics, Metrics


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0]
    y[:, 1] = x[:, 1]
    y[:, 2] = x[:, 0] + x[:, 2]  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3]  # bottom right y
    return y


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def bbox_overlaps(boxes, query_boxes):
    n_ = boxes.shape[0]
    k_ = query_boxes.shape[0]
    overlaps = np.zeros((n_, k_), dtype=np.float)
    for k in range(k_):
        query_box_area = (query_boxes[k, 2] - query_boxes[k, 0] + 1) * (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        for n in range(n_):
            iw = min(boxes[n, 2], query_boxes[k, 2]) - max(boxes[n, 0], query_boxes[k, 0]) + 1
            if iw > 0:
                ih = min(boxes[n, 3], query_boxes[k, 3]) - max(boxes[n, 1], query_boxes[k, 1]) + 1
                if ih > 0:
                    box_area = (boxes[n, 2] - boxes[n, 0] + 1) * (boxes[n, 3] - boxes[n, 1] + 1)
                    all_area = float(box_area + query_box_area - iw * ih)
                    overlaps[n, k] = iw * ih / all_area
    return overlaps


def bbox_overlap(a, b):
    x1 = np.maximum(a[:, 0], b[0])
    y1 = np.maximum(a[:, 1], b[1])
    x2 = np.minimum(a[:, 2], b[2])
    y2 = np.minimum(a[:, 3], b[3])
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    inter = w * h
    aarea = (a[:, 2] - a[:, 0] + 1) * (a[:, 3] - a[:, 1] + 1)
    barea = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
    o = inter / (aarea + barea - inter)
    o[w <= 0] = 0
    o[h <= 0] = 0
    return o


def __bbox_overlap(a, b):
    x1 = torch.max(a[:, 0], b[0])
    y1 = torch.max(a[:, 1], b[1])
    x2 = torch.min(a[:, 2], b[2])
    y2 = torch.min(a[:, 3], b[3])
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    inter = w * h
    aarea = (a[:, 2] - a[:, 0] + 1) * (a[:, 3] - a[:, 1] + 1)
    barea = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
    o = inter / (aarea + barea - inter)
    o[w <= 0] = 0
    o[h <= 0] = 0
    return o


def np_around(array, num_decimals=0):
    # return array
    return np.around(array, decimals=num_decimals)


def np_round(val, decimals=4):
    return val


def get_gt_boxes(gt_dir):
    """ gt dir: (wider_face_val.mat, wider_easy_val.mat, wider_medium_val.mat, wider_hard_val.mat)"""

    gt_mat = loadmat(os.path.join(gt_dir, 'wider_face_val.mat'))
    hard_mat = loadmat(os.path.join(gt_dir, 'wider_hard_val.mat'))
    medium_mat = loadmat(os.path.join(gt_dir, 'wider_medium_val.mat'))
    easy_mat = loadmat(os.path.join(gt_dir, 'wider_easy_val.mat'))

    facebox_list = gt_mat['face_bbx_list']
    event_list = gt_mat['event_list']
    file_list = gt_mat['file_list']

    hard_gt_list = hard_mat['gt_list']
    medium_gt_list = medium_mat['gt_list']
    easy_gt_list = easy_mat['gt_list']

    return facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list


def get_gt_boxes_from_txt(gt_path, cache_dir):
    cache_file = os.path.join(cache_dir, 'gt_cache.pkl')
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            boxes = pickle.load(f)
        return boxes

    f = open(gt_path, 'r')
    state = 0
    lines = f.readlines()
    lines = list(map(lambda x: x.rstrip('\r\n'), lines))
    boxes = {}
    # print(len(lines))
    f.close()
    current_boxes = []
    current_name = None
    for line in lines:
        if state == 0 and '--' in line:
            state = 1
            current_name = line
            continue
        if state == 1:
            state = 2
            continue

        if state == 2 and '--' in line:
            state = 1
            boxes[current_name] = np.array(current_boxes).astype('float32')
            current_name = line
            current_boxes = []
            continue

        if state == 2:
            box = [float(x) for x in line.split(' ')[:4]]
            current_boxes.append(box)
            continue

    with open(cache_file, 'wb') as f:
        pickle.dump(boxes, f)
    return boxes


def read_pred_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        img_file = lines[0].rstrip('\n\r')
        lines = lines[2:]

    boxes = np.array(list(map(lambda x: [float(a) for a in x.rstrip('\r\n').split(' ')], lines))).astype('float')
    return img_file.split('/')[-1], boxes


def get_preds(pred_dir):
    events = os.listdir(pred_dir)
    boxes = dict()
    pbar = tqdm.tqdm(events)

    for event in pbar:
        pbar.set_description('Reading Predictions ')
        event_dir = os.path.join(pred_dir, event)
        event_images = os.listdir(event_dir)
        current_event = dict()
        for imgtxt in event_images:
            imgname, _boxes = read_pred_file(os.path.join(event_dir, imgtxt))
            current_event[imgname.rstrip('.jpg')] = _boxes
        boxes[event] = current_event
    return boxes


def norm_score(pred):
    """ norm score
    pred {key: [[x1,y1,x2,y2,s]]}
    """

    max_score = -1
    min_score = 2

    for _, k in pred.items():
        for _, v in k.items():
            if len(v) == 0:
                continue
            _min = np.min(v[:, -1])
            _max = np.max(v[:, -1])
            max_score = max(_max, max_score)
            min_score = min(_min, min_score)

    diff = max_score - min_score
    for _, k in pred.items():
        for _, v in k.items():
            if len(v) == 0:
                continue
            v[:, -1] = (v[:, -1] - min_score).astype(np.float64) / diff
    return pred


def image_eval(pred, gt, ignore, iou_thresh):
    """ single image evaluation
    pred: Nx5
    gt: Nx4
    ignore:
    """

    _pred = pred.copy()
    _gt = gt.copy()

    _pred[:, :4] = xywh2xyxy(_pred[:, :4])
    _gt[:, :4] = xywh2xyxy(_gt[:, :4])

    iou_matrix = box_iou(torch.from_numpy(_pred[:, :4]), torch.from_numpy(_gt[:, :4])).numpy()

    pred_recall = np.zeros(_pred.shape[0])
    recall_list = np.zeros(_gt.shape[0])
    proposal_list = np.ones(_pred.shape[0])
    for pred_ind in range(_pred.shape[0]):
        gt_overlap = iou_matrix[pred_ind]

        max_overlap, max_idx = gt_overlap.max(), gt_overlap.argmax()

        if max_overlap >= iou_thresh:
            if ignore[max_idx] == 0:
                recall_list[max_idx] = -1
                proposal_list[pred_ind] = -1
            elif recall_list[max_idx] == 0:
                recall_list[max_idx] = 1

        r_keep_index = np.where(recall_list == 1)[0]
        pred_recall[pred_ind] = len(r_keep_index)

    return pred_recall, proposal_list


def img_pr_info(thresh_num, pred_info, proposal_list, pred_recall):
    pr_info = np.zeros((thresh_num, 2)).astype('float')
    fp = np.zeros((pred_info.shape[0],), dtype=np.int32)
    last_info = [-1, -1]
    for t in range(thresh_num):

        thresh = 1 - (t + 1) / thresh_num
        r_index = np.where(pred_info[:, 4] >= thresh)[0]
        if len(r_index) == 0:
            pr_info[t, 0] = 0
            pr_info[t, 1] = 0
        else:
            r_index = r_index[-1]
            p_index = np.where(proposal_list[:r_index + 1] == 1)[0]
            pr_info[t, 0] = len(p_index)  # valid pred number
            pr_info[t, 1] = pred_recall[r_index]  # valid gt number

            if t > 0 and pr_info[t, 0] > pr_info[t - 1, 0] and pr_info[t, 1] == pr_info[t - 1, 1]:
                fp[r_index] = 1
                # if thresh>=0.85:
                #    print(thresh, t, pr_info[t])
    # print(pr_info[:10,0])
    # print(pr_info[:10,1])
    return pr_info, fp


def dataset_pr_info(thresh_num, pr_curve, count_face):
    _pr_curve = np.zeros((thresh_num, 2))
    for i in range(thresh_num):
        # _pr_curve[i, 0] = round(pr_curve[i, 1] / pr_curve[i, 0], 4)
        # _pr_curve[i, 1] = round(pr_curve[i, 1] / count_face, 4)
        _pr_curve[i, 0] = pr_curve[i, 1] / pr_curve[i, 0]
        _pr_curve[i, 1] = pr_curve[i, 1] / count_face
    return _pr_curve


def voc_ap(rec, prec):
    # correct AP calculation
    # first append sentinel values at the end
    # print ('rec:', rec)
    # print ('pre:', prec)
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np_round(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))
    return ap


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0]
    y[:, 1] = x[:, 1]
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= 0 #pad[0]  # x padding     # TODO
    coords[:, [1, 3]] -= 0 #pad[1]  # y padding
    coords[:, :4] /= gain

    clip_coords(coords, img0_shape)
    return coords


class WiderFaceEvaluator:
    thresh_num = 1000
    settings = ['easy', 'medium', 'hard']

    def __init__(self, gt_dir, iou_thresh=0.5):
        self.gt_dir = gt_dir
        self.iou_thresh = iou_thresh

        gt_data = get_gt_boxes(gt_dir)
        self.facebox_list, self.event_list, self.file_list, hard_gt_list, medium_gt_list, easy_gt_list = gt_data
        self.event_num = len(self.event_list)
        self.setting_gts = [easy_gt_list, medium_gt_list, hard_gt_list]

        self.pred = {}

    def add_batch(self, predictions, targets, meta_data):
        img_paths = meta_data['img_path']
        img0_shapes = meta_data['img0_shape']
        img1_shape = meta_data['img1_shape']

        scores_pred, labels_pred, bbox_pred = predictions[:3]
        batch_size = len(img_paths)

        for i in range(batch_size):
            scores, labels = scores_pred[i], labels_pred[i]
            boxes = bbox_pred[i]
            img0_shape, img1_shape, img_path = img0_shapes[i], img1_shape, img_paths[i]

            event_name, name = img_path.split('/')[-2:]
            name = name.split('.')[0]

            if len(boxes) == 0:
                p = []
            else:
                boxes = scale_coords(img1_shape, boxes, img0_shape, None)
                clip_coords(boxes, img0_shape)
                boxes = xyxy2xywh(boxes)

                p = np.empty((len(boxes), 5), dtype=np.float32)
                for i, (s, l, bb) in enumerate(zip(scores, labels, boxes)):
                    bb = bb.to('cpu').numpy()
                    p[i] = np.array([bb[0], bb[1], bb[2], bb[3], s.to('cpu').numpy()], dtype=np.float32)

            if event_name not in self.pred:
                self.pred[event_name] = {name: p}
            else:
                self.pred[event_name][name] = p

    def compute(self) -> Dict:
        metrics = {}
        for setting_id in range(len(self.settings)):
            gt_list = self.setting_gts[setting_id]
            count_face = 0
            pr_curve = np.zeros((self.thresh_num, 2)).astype('float')
            # [hard, medium, easy]
            for i in range(self.event_num):
                event_name = str(self.event_list[i][0][0])
                img_list = self.file_list[i][0]
                pred_list = self.pred[event_name]
                sub_gt_list = gt_list[i][0]
                gt_bbx_list = self.facebox_list[i][0]

                for j in range(len(img_list)):
                    img_name = str(img_list[j][0][0])
                    pred_info = pred_list[img_name]

                    gt_boxes = gt_bbx_list[j][0].astype('float')
                    keep_index = sub_gt_list[j][0]
                    count_face += len(keep_index)

                    if len(gt_boxes) == 0 or len(pred_info) == 0:
                        continue

                    ignore = np.zeros(gt_boxes.shape[0], dtype=np.int32)
                    if len(keep_index) != 0:
                        ignore[keep_index - 1] = 1
                    pred_info = np_round(pred_info, 1)

                    gt_boxes = np_round(gt_boxes)
                    pred_recall, proposal_list = image_eval(pred_info, gt_boxes, ignore, self.iou_thresh)

                    _img_pr_info, fp = img_pr_info(self.thresh_num, pred_info, proposal_list, pred_recall)

                    pr_curve += _img_pr_info

            pr_curve = dataset_pr_info(self.thresh_num, pr_curve, count_face)

            precision = pr_curve[:, 0]
            recall = pr_curve[:, 1]
            # for f in range(thresh_num):
            #    print('R-P:', recall[f], propose[f])
            # for srecall in np.arange(0.1, 1.0001, 0.1):
            #     rindex = len(np.where(recall <= srecall)[0]) - 1
            #     rthresh = 1.0 - float(rindex) / self.thresh_num
            #     print('Recall-Precision-Thresh:', recall[rindex], precision[rindex], rthresh)

            ap = voc_ap(recall, precision)
            # aps[setting_id] = ap
            # tb = datetime.datetime.now()
            # print('high score count:', high_score_count)
            # print('high score fp count:', high_score_fp_count)
            # print('%s cost %.4f seconds, ap: %.5f' % (self.settings[setting_id], (tb - ta).total_seconds(), ap))

            metrics[f'{self.settings[setting_id]}_AP'] = ap

        out_metrics = DSMetrics()
        out_metrics.extend_with(bb=metrics)
        self.metrics = out_metrics
        return out_metrics

    def reset(self):
        self.pred = {}
