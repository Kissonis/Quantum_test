import glob
import random
import os
import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


class TrainImageDataset(Dataset):
    def __init__(self, root, hr_shape):
        hr_height, hr_width = hr_shape
        # Transforms for images
        self.imgs_transform = transforms.Compose(
            [
                transforms.Resize((hr_height, hr_height), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.masks_transform = transforms.Compose(
            [
                transforms.Resize((hr_height, hr_height)),
                transforms.ToTensor(),
                #transforms.Normalize(mean, std),
            ]
        )

        self.files_imgs = sorted(glob.glob(root + "/img_train/images/*.*"))
        self.files_masks = sorted(glob.glob(root + "/img_train/masks/*.*"))

    def __getitem__(self, index):
        imgs = Image.open(self.files_imgs[index % len(self.files_imgs)])
        imgs = self.imgs_transform(imgs)
        masks = Image.open(self.files_masks[index % len(self.files_masks)])
        masks = self.masks_transform(masks)

        return {"images": imgs, "masks": masks}

    def __len__(self):
        return len(self.files_imgs)

class TestImageDataset(Dataset):
    def __init__(self, root, hr_shape):
        hr_height, hr_width = hr_shape
        # Transforms for images
        self.imgs_transform = transforms.Compose(
            [
                transforms.Resize((hr_height, hr_height), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        self.files_imgs = sorted(glob.glob(root + "/img_test/images/*.*"))

    def __getitem__(self, index):
        imgs = Image.open(self.files_imgs[index % len(self.files_imgs)])
        imgs = self.imgs_transform(imgs)

        return {"images": imgs}

    def __len__(self):
        return len(self.files_imgs)