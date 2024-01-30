import cv2
import warnings
import numpy as np
from PIL import Image
import torch


try:
    from turbojpeg import TurboJPEG
    JPEG_READER = TurboJPEG()
except ImportError:
    JPEG_READER = None


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0]
    y[:, 1] = x[:, 1]
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0]
    y[:, 1] = x[:, 1]
    y[:, 2] = x[:, 0] + x[:, 2]  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3]  # bottom right y
    return y


def list_xyxy2xywh(box):
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]


def list_xywh2xyxy(box):
    x1, y1, w, h = box
    return [x1, y1, x1 + w, y1 + h]


def load_image(img_path, engine, color_format) -> np.ndarray:
    if engine == 'turbojpeg' and JPEG_READER is None:
        raise RuntimeError('TurboJPEG is not installed to be used as engine')

    if engine == 'cv2':
        img = cv2.imread(img_path)  # BGR
        if color_format.lower() == 'rgb':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img, dtype=np.uint8)
    elif engine == 'pil':
        img = np.array(Image.open(img_path))
        if color_format.lower() == 'bgr':
            img = img[:, :, ::-1]
    elif engine == 'turbojpeg':
        in_file = open(img_path, 'rb')
        img = np.array(JPEG_READER.decode(in_file.read()))
        in_file.close()
        if color_format.lower() == 'bgr':
            img = img[:, :, ::-1]
    else:
        raise NotImplementedError('Unknown engine supplied. Available: [pil, cv2, turbojpeg]')
    assert img is not None, f'Image Not Found {img_path}'

    return img
