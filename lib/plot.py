import cv2
import math
import torch
import numpy as np
from datasets.utils import xyxy2xywh


def plot_predictions(images, predictions, save_path=None, max_size=640, max_subplots=32):
    b, c, h, w = images.size()
    scores, labels, bboxes, keypoints = predictions

    b = min(b, max_subplots)  # limit plot images
    ns = np.ceil(b ** 0.5)  # number of subplots (square)

    # Check if we should resize
    scale_factor = max_size / max(h, w)
    if scale_factor < 1:
        h = math.ceil(scale_factor * h)
        w = math.ceil(scale_factor * w)

    # Empty array for output
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)

    for i in range(b):
        if i == max_subplots:  # if last batch has fewer images than we expect
            break

        img = images.to('cpu').numpy()[i]
        img = np.transpose(img, (1, 2, 0))

        block_x = int(w * (i // ns))
        block_y = int(h * (i % ns))

        if scale_factor < 1:
            img = cv2.resize(img, (w, h))

        scs = scores[i]
        lbl = labels[i]
        bbs = bboxes[i]
        kps = keypoints[i]

        if len(bbs) == 0:
            mosaic[block_y:block_y + h, block_x:block_x + w, :] = img
            continue

        annotations = predictions2coco(scs, lbl, bbs, kps, img_id=-1)

        mosaic[block_y:block_y + h, block_x:block_x + w, :] = draw_face_det(img, annotations)

    if save_path:
        cv2.imwrite(save_path, mosaic)

    return mosaic


def predictions2coco(scores, labels, bboxes, keypoints, img_id, ann_counter_start=0):
    coco_annotations = []
    bboxes = xyxy2xywh(torch.stack(bboxes))

    for score, label, bb, kps in zip(scores, labels, bboxes, keypoints):
        box = bb.tolist()
        kp = kps.tolist()

        coco_kp = []
        for i in range(len(kp) // 2):
            point = [*kp[2*i:2*(i+1)], 1.0]
            coco_kp.extend(point)
        kp = coco_kp
        ann = {
            'id': ann_counter_start,  # will be overridden during tmp json annotation creation while compute()
            'image_id': img_id,

            # bbox
            'category_id': label,
            'bbox': box,
            'area': box[2] * box[3],

            # keypoints
            'keypoints': kp,
            'num_keypoints': len(kp) // 3 if kp is not None else -1,

            # tags:
            'iscrowd': 0,  # default coco tag (always constant for our tasks)

            'score': score
        }
        coco_annotations.append(ann)
        ann_counter_start += 1
    return coco_annotations


def denormalize_image(image, mean, std):
    device = image.device

    mean = torch.tensor(mean).view(1, 3, 1, 1).to(device)
    std = torch.tensor(std).view(1, 3, 1, 1).to(device)

    if (mean <= 1.0).all():
        mean *= 255
    if (std <= 1.0).all():
        std *= 255
    out = torch.clip(image * std + mean, 0.0, 255.0)
    return out

def str2color(color: str):
    color_map = {
        'red': (1.0, 0.0, 0.0),
        'cyan': (0.0, 1.0, 1.0),
        'black': (0.0, 0.0, 0.0),
        'green': (0.0, 1.0, 0.0),
        'purple': (0.5, 0.0, 0.5),
    }
    color = np.array(color_map[color])
    color *= 255
    color = list(map(int, color))
    return color


def list_xywh2xyxy(box):
    x, y, w, h = box
    return [x, y, x+w, y+h]


def draw_box(img, coco_bbox, color='green', fill=False, line_width=1):
    x1, y1, x2, y2 = map(int, list_xywh2xyxy(coco_bbox))
    if fill:
        line_width *= -1

    color = str2color(color)
    out_img = cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=line_width)
    return out_img


def draw_key_points(img, coco_kps, color='auto', line_width=1, marker_size=8, draw_labels=False):
    out_img = img.copy()

    for ind in range(len(coco_kps) // 3):
        start_ind = 3*ind
        end_ind = 3*(ind+1)
        x, y, v = map(int, coco_kps[start_ind:end_ind])

        if color == 'auto':
            if v == 2:
                kp_color = str2color('cyan')
            elif v == 1:
                kp_color = str2color('purple')
            else:
                kp_color = str2color('black')
        else:
            kp_color = str2color(color)

        out_img = cv2.circle(out_img, center=(x, y), radius=marker_size, color=kp_color, thickness=-1)

        if draw_labels:
            font = cv2.FONT_HERSHEY_SIMPLEX
            out_img = cv2.putText(out_img, text=str(ind), org=(x, y), fontFace=font, fontScale=0.8, color=kp_color, thickness=line_width)

    return out_img


def draw_face_det(img, annotations):
    height, width, _ = img.shape

    out_img = img.copy()
    for ann in annotations:
        min_side = min(ann['bbox'][2], ann['bbox'][3])
        marker_size = max(1, int(min_side / 20))

        box = ann['bbox']

        out_img = draw_box(out_img, box, color='green')

        # lbl_text = f'Face: {ann["score"]:.2f}'
        # draw_text(ax, lbl_text, lbl_pos, font_size=10, color=color)

        kps = ann.get('keypoints')
        if kps:
            out_img = draw_key_points(out_img, ann['keypoints'], color='auto', marker_size=marker_size, draw_labels=False)
        # draw_text(ax, id_text, id_pos, font_size=8, color=color)
    return out_img

