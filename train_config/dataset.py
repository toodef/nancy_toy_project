import json
import os
from typing import List, Tuple

import cv2
import numpy as np
import torch
from albumentations import Compose, GaussNoise, RandomGamma, RandomBrightnessContrast, HorizontalFlip, SmallestMaxSize, RandomCrop, \
    OneOf, Resize, CenterCrop

from pietoolbelt.datasets.common import BasicDataset
from pietoolbelt.datasets.utils import AugmentedDataset

from config import DATASET_LABELS, DATASET_ROOT
from train_config.config import data_height, data_width


class CatsDogsDataset(BasicDataset):
    def __init__(self):
        items = []
        for file in os.listdir(os.path.join(DATASET_ROOT, 'images_original')):
            items.append(os.path.join(DATASET_ROOT, 'images_original', file))

        items.sort(key=lambda x: os.path.splitext(x)[0])
        super().__init__(items)

    def _interpret_item(self, item) -> any:
        return {'data': np.zeros((512, 521, 3), dtype=np.uint8), 'target': 0}


class Augmentations:
    def __init__(self, is_train: bool, to_pytorch: bool):
        if is_train:
            self._aug = Compose([
                OneOf([Compose([SmallestMaxSize(max_size=min(data_height, data_width) * 1.1, p=1),
                                RandomCrop(height=data_height, width=data_width, p=1)], p=1),
                       Resize(height=data_height, width=data_width, p=1)], p=1),
                GaussNoise(p=0.5),
                RandomGamma(p=0.5),
                RandomBrightnessContrast(p=0.5),
                HorizontalFlip(p=0.5)
            ], p=1)
        else:
            self._aug = Compose([SmallestMaxSize(max_size=min(data_height, data_width), p=1),
                                 CenterCrop(height=data_height, width=data_width, p=1)], p=1)

        self._need_to_pytorch = to_pytorch

    def augmentate(self, data: {}):
        if self._aug is not None:
            augmented = self._aug(image=data['data'], mask=data['target'])
            img, mask = augmented['image'], augmented['mask']
        else:
            img, mask = data['data'], data['target']

        if self._need_to_pytorch:
            img = self.img_to_pytorch(img)
            mask = self.target_to_pytorch(mask)

        return {'data': img, 'target': mask}

    @staticmethod
    def img_to_pytorch(image):
        return torch.from_numpy(np.moveaxis(image, -1, 0).astype(np.float32) / 128 - 1)

    @staticmethod
    def target_to_pytorch(target):
        return torch.from_numpy(np.expand_dims(target.astype(np.float32), 0) / (target.max() + 1e-7))


def create_dataset(indices_path: str = None) -> 'LabeledDataset':
    dataset = CatsDogsDataset()

    if indices_path is not None:
        dataset.load_indices(indices_path).remove_unused_data()
    return dataset


def create_augmented_dataset(is_train: bool, to_pytorch: bool = True, indices_path: str = None) -> 'AugmentedDataset':
    dataset = create_dataset(indices_path)
    augs = Augmentations(is_train, to_pytorch)

    # return AugmentedDataset(DebugDataset(dataset, 20)).add_aug(augs.augmentate)
    return AugmentedDataset(dataset).add_aug(augs.augmentate)
