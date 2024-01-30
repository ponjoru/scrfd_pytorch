import json
import albumentations as A

from loguru import logger
from pathlib import Path
from multiprocessing import Manager
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from typing import Optional, Callable


from datasets.utils import load_image


class BaseDataset(ABC, Dataset):
    """
    Base class for all the datasets

    Args:
        ds_path (str): Path to the dataset root
        split (str): train, val, or test
        transforms (Optional[albumentations.Compose]): data transforms based on albumentations
        color_layout (str): RGB, BGR, or any supported type in subclasses
        engine (str): library used to read images, default = 'cv2'
        cache_images (bool): preliminary load all the images in the dataset to RAM 
    """
    def __init__(
            self,
            ds_path: str,
            split: str,
            transforms: Optional[A.Compose] = None,
            color_layout: str = 'RGB',
            engine: str = 'cv2',
            cache_images: bool = False
    ):
        self.ds_path = Path(ds_path)
        self.split = split
        self.transforms = transforms
        self.color_layout = color_layout.lower()
        self.engine = engine.lower()
        self.cache_images = cache_images

        self.images_meta, self.annotations = self._load_meta_data()

        manager = Manager()
        self.images = list() if not self.cache_images else self._load_images_to_memory()
        self.annotations = manager.list(self.annotations)

    def _load_img_dict(self, index):
        img_meta = self.images_meta[index]
        img_path = str(self.ds_path / img_meta['img_path'])

        img = load_image(img_path, self.engine, self.color_layout)

        h0, w0 = img.shape[:2]  # orig hw

        img_dict = {
            'image': img,
            'img_path': img_path,
            'img_id': img_meta['img_id'],
            'img_shape': (h0, w0)
        }
        return img_dict

    def _load_image(self, index):
        if self.cache_images:
            return self.images[index]
        else:
            return self._load_img_dict(index)

    def _load_images_to_memory(self):
        for i in range(len(self.images_meta)):
            data = self._load_img_dict(index=i)
            self.images.append(data)

    def __len__(self):
        return len(self.images_meta)

    def __getitem__(self, index):
        """ BBOX format: XYXY """
        img_dict = self._load_image(index)
        annotations = self.annotations[index]

        if self.transforms:
            img_dict, annotations = self._transform_wrapper(self.transforms, img_dict, annotations)

        ann_dict = self._create_ann_dict(annotations)
        sample = {
            **img_dict,
            **ann_dict,
        }
        return sample

    @property
    def get_ds_name(self):
        return str(Path(self.ds_path).name)

    @staticmethod
    @abstractmethod
    def collate_fn(batch):
        raise NotImplementedError

    @abstractmethod
    def _load_meta_data(self):
        raise NotImplementedError

    @abstractmethod
    def _transform_wrapper(self, transforms, img_dict, annotations):
        raise NotImplementedError

    @abstractmethod
    def _create_ann_dict(self, annotations):
        raise NotImplementedError
