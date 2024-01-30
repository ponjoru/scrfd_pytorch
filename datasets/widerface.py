import os
import torch
import numpy as np
import albumentations as A
from pathlib import Path


from datasets.utils import xyxy2xywh, xywh2xyxy, list_xywh2xyxy
from datasets.base_dataset import BaseDataset


class SimpleCustomBatch:
    def __init__(self, data):
        bs = len(data)
        img_size = data[0]['image'].size()  # CHW

        self.image = torch.zeros((bs, *img_size))
        self.annotations = [x['annotations'] for x in data]
        for i, b in enumerate(self.annotations):
            self.annotations[i][:, 0] = i  # add target image index
        self.annotations = torch.cat(self.annotations)

        self.meta = {
            'img_path': [],
            'img_id': [],
            'img0_shape': [],
            'img1_shape': list(img_size[1:]),
        }

        for i, item in enumerate(data):
            self.image[i] = item['image']
            self.meta['img_path'].append(item['img_path'])
            self.meta['img_id'].append(item['img_id'])
            self.meta['img0_shape'].append(item['img_shape'])

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.image = self.image.pin_memory()
        self.annotations = self.annotations.pin_memory()
        return self

    def to(self, device, channels_last=False):
        self.image = self.image.to(device)
        self.annotations = self.annotations.to(device)

        if channels_last:
            self.image = self.image.to(memory_format=torch.channels_last)

    def __getitem__(self, item):
        return getattr(self, item)


def is_in_image(point, shape):
    return 0 <= point[0] < shape[0] and 0 <= point[1] < shape[1]


class WiderFaceDataset(BaseDataset):
    NK = 5
    BB_CLASS_LABELS = ('Face', )
    KP_CLASS_LABELS = ['left_eye', 'right_eye', 'nose', 'left_mouth', 'right_mouth']

    def __init__(self, ds_path, split, min_size=None, transforms=None, color_layout='RGB', engine='cv2', cache_images=False):
        self.min_size = min_size
        self.bb_cat2id = {cat: idx for idx, cat in enumerate(self.BB_CLASS_LABELS)}
        self.bb_id2cat = {idx: cat for idx, cat in enumerate(self.BB_CLASS_LABELS)}

        self.kp_cat2id = {cat: idx for idx, cat in enumerate(self.KP_CLASS_LABELS)}
        self.kp_id2cat = {idx: cat for idx, cat in enumerate(self.KP_CLASS_LABELS)}

        super(WiderFaceDataset, self).__init__(ds_path, split, transforms, color_layout, engine, cache_images)

    def _load_meta_data(self):
        img_meta = []
        ann_meta = []

        name = None
        bbox_map = {}

        ann_path = str(Path(self.ds_path) / self.split / 'labelv2.txt')
        for line in open(ann_path, 'r'):
            line = line.strip()
            if line.startswith('#'):
                value = line[1:].strip().split()
                name = value[0]
                width = int(value[1])
                height = int(value[2])

                bbox_map[name] = dict(width=width, height=height, objs=[])
                continue

            assert name is not None
            assert name in bbox_map
            bbox_map[name]['objs'].append(line)

        data_infos = []
        for name in bbox_map:
            item = bbox_map[name]
            width = item['width']
            height = item['height']
            vals = item['objs']

            objs = []
            for line in vals:
                data = self._parse_ann_line(line)
                if data is None:
                    continue
                objs.append(data)   # data is (bbox, kps, cat)

            # if len(objs) == 0:
            #     continue

            data_infos.append(dict(filename=name, width=width, height=height, objs=objs))

        for ind, info in enumerate(data_infos):
            objects = info['objs']

            ann_list = []
            for idx, obj in enumerate(objects):
                bb = obj['bbox']
                cls_id = self.bb_cat2id[obj['cat']]
                kp = np.array(obj['kps']).reshape(-1)

                ann_list.append([0.0, cls_id, 1.0, *bb, *kp])

            img_shape = (info['height'], info['width'])
            annotations = np.array(ann_list, dtype=np.float32).reshape(-1, 22)
            annotations[:, 3:7], keep = self.validate_bb(annotations[:, 3:7], img_shape=img_shape)
            annotations = annotations[keep]
            annotations[:, 7:] = self.validate_kp(annotations[:, 7:], annotations[:, 3:7])

            img_path = str(Path(self.ds_path) / self.split / 'images' / info['filename'])
            img_meta.append({
                'img_id': ind,
                'img_path': img_path,
                'img_shape': img_shape,
                'tags': {}
            })
            ann_meta.append({'annotations': annotations})

        return img_meta, ann_meta

    def _transform_wrapper(self, transforms, img_dict, annotations):
        annotations = annotations['annotations']
        n = len(annotations)
        kps = annotations[:, 7:].reshape(-1, 3)

        bbox_id = np.arange(n)
        kp_labels = np.array([self.kp_cat2id[_] for _ in self.KP_CLASS_LABELS])
        kp_labels = kp_labels.reshape(1, self.NK).repeat(n, axis=0).flatten()
        kp2bb_ids = np.array(list(range(n))).repeat(self.NK)

        transformed = transforms(
            image=img_dict['image'],
            bboxes=annotations[:, 3:7],
            bb_classes=annotations[:, 1],
            bb_weights=annotations[:, 2],
            bbox_id=bbox_id,
            keypoints=kps[:, :2],
            kp_classes=kp_labels,
            kp_weights=kps[:, 2],
            kp2bb_ids=kp2bb_ids
        )

        img_dict['image'] = transformed['image']

        out_annotations = np.zeros((len(transformed['bboxes']), 22), dtype=np.float32)

        if len(transformed['bboxes']):
            out_annotations[:, 1] = np.array(transformed['bb_classes'], dtype=np.float32)
            out_annotations[:, 2] = np.array(transformed['bb_weights'], dtype=np.float32)
            out_annotations[:, 3:7] = np.array(transformed['bboxes'], dtype=np.float32).reshape(-1, 4)

            bbox_id = transformed['bbox_id']
            _id_map = {bbox_id[i]: i for i in range(len(out_annotations))}

            kp, kp_weights, kp2bb_ids = transformed['keypoints'], transformed['kp_weights'], transformed['kp2bb_ids']
            kp_labels = transformed['kp_classes']

            for i, (point, label, w, kp2bb_id) in enumerate(zip(kp, kp_labels, kp_weights, kp2bb_ids)):
                lbl_id = i % 5
                weight = w if is_in_image(point, transformed['image'].size()[-2:]) else 0.0

                start_ind = 7 + 3 * lbl_id
                end_ind = 7 + 3 * (lbl_id + 1)
                box_id = _id_map.get(kp2bb_id, None)

                if box_id is not None:
                    out_annotations[box_id, start_ind:end_ind] = [*point, weight]

        annotations = {'annotations': out_annotations}
        return img_dict, annotations

    @staticmethod
    def _create_ann_dict(annotations):
        annotations = torch.tensor(annotations['annotations'], dtype=torch.float)

        ann_dict = {
            'annotations': annotations,
        }
        return ann_dict

    def _parse_ann_line(self, line):
        values = [float(x) for x in line.strip().split()]
        bbox = np.array(values[0:4], dtype=np.float32)
        kps = np.zeros((self.NK, 3), dtype=np.float32)
        ignore = False
        if self.min_size is not None:
            assert not self.test_mode
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            if w < self.min_size or h < self.min_size:
                ignore = True
        if len(values) > 4:
            if len(values) > 5:
                kps = np.array(values[4:19], dtype=np.float32).reshape((self.NK, 3))
                for li in range(kps.shape[0]):
                    if (kps[li, :] == -1).all():
                        kps[li][2] = 0.0    # weight = 0, ignore
                    else:
                        assert kps[li][2] >= 0
                        kps[li][2] = 1.0    # weight
            else:
                if not ignore:
                    ignore = (values[4] == 1)

        return dict(bbox=bbox, kps=kps, ignore=ignore, cat='Face')

    @staticmethod
    def validate_bb(bb, img_shape):
        h, w = img_shape
        bb[:, 0::2] = bb[:, 0::2].clip(0, w - 1)
        bb[:, 1::2] = bb[:, 1::2].clip(0, h - 1)
        keep = ((bb[:, 0] != bb[:, 2]) * (bb[:, 1] != bb[:, 3])).nonzero()
        return bb, keep

    @staticmethod
    def validate_kp(kp, bb):
        """
        Clip key points to the bbox size. If kp coords differ more than by 1 pixel: kp is marked as ignore
        bb: np.array(nl, 4)
        kp: np.array(nl, 15)
        """
        kp = kp.reshape(-1, 5, 3)
        for i in range(len(kp)):
            box = bb[i]

            x1, y1, x2, y2 = box

            # set outbound kps as ignored
            x_ignore_1 = kp[i, :, 0] < x1 - 1
            x_ignore_2 = kp[i, :, 0] > x2
            y_ignore_1 = kp[i, :, 1] < y1 - 1
            y_ignore_2 = kp[i, :, 1] > y2
            weight = np.sum(np.stack([x_ignore_1, x_ignore_2, y_ignore_1, y_ignore_2]), axis=0) == 0
            kp[i, :, 2] = weight

            # clip all key points to the bbox size
            kp[i, :, 0] = kp[i, :, 0].clip(x1, x2 - 1)
            kp[i, :, 1] = kp[i, :, 1].clip(y1, y2 - 1)

        kp = kp.reshape(-1, 15)
        return kp

    @property
    def get_ds_name(self):
        return str(Path(self.ds_path).name)

    @staticmethod
    def collate_fn(data):
        batch = SimpleCustomBatch(data)
        return batch


if __name__ == '__main__':
    import cv2
    import albumentations as A
    import albumentations_experimental
    from albumentations.pytorch.transforms import ToTensorV2

    from lib.utils.torch_utils import denormalize_image
    from detpack.transforms.custom_alb import BBSafeScaledRandomCrop

    mean = [0.5, 0.5, 0.5]
    std = [0.50196, 0.50196, 0.50196]
    transforms = A.Compose([
        BBSafeScaledRandomCrop(p=1.0),
        albumentations_experimental.HorizontalFlipSymmetricKeypoints(symmetric_keypoints=[[0, 1], [2, 2], [3, 4]], p=1.0),
        # A.ColorJitter(hue=0.015, saturation=0.7, contrast=0.2, brightness=0.4, p=1.0),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ],
        bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.7, label_fields=['bb_classes', 'bb_weights', 'bbox_id']),
        keypoint_params=A.KeypointParams(format='xy', label_fields=['kp_classes', 'kp_weights', 'kp2bb_ids'], remove_invisible=False)
    )

    ds = WiderFaceDataset(
        ds_path='/datasets/face_detection/widerface/widerface',
        split='train',
        transforms=transforms,
        color_layout='BGR',
    )

    data = ds[0]

    def draw_box(img, bbox, color='green', fill=False, line_width=1):
        x1, y1, x2, y2 = bbox
        if fill:
            line_width *= -1

        color = (255.0, 255.0, 0.0)
        out_img = cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=line_width)
        return out_img


    def draw_kps(img, kps, marker_size=2, draw_labels=True):
        out_img = img.copy()
        mod_ = 3
        for ind in range(len(kps) // mod_):
            start_ind = mod_ * ind
            end_ind = mod_ * (ind + 1)
            x, y, v = map(int, kps[start_ind:end_ind])
            if v == 1:
                kp_color = (255, 0, 0)
            elif v == -1:
                kp_color = (0, 255, 0)
            elif v == 0:
                kp_color = (0, 0, 255)

            out_img = cv2.circle(out_img, center=(x, y), radius=marker_size, color=kp_color, thickness=-1)

            if draw_labels:
                font = cv2.FONT_HERSHEY_SIMPLEX
                out_img = cv2.putText(out_img, text=str(ind), org=(x, y), fontFace=font, fontScale=0.8, color=kp_color,
                                      thickness=1)

        return out_img

    from lib.utils.torch_utils import denormalize_image
    if isinstance(data['image'], np.ndarray):
        img = data['image']
    else:
        img = denormalize_image(data['image'].unsqueeze(0), mean, std)[0]
        img = np.transpose(img.numpy(), (1, 2, 0))

    for box in data['annotations'].numpy().astype(np.int32):
        # print(box[2:6])
        # print(box[7:])
        img = draw_box(img, box[2:6])
        img = draw_kps(img, box[7:])

    cv2.imwrite('/app/example.jpg', img)

    k = 5