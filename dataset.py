import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

SPLIT_CHARACTER = "_"
MASK_COLOR_MAP = "L"
MASK_BASE_NAME = "mask"


def normalize_array(array):
    """"""
    dividend = array - np.min(array)
    divisor = np.max(array) - np.min(array)
    if divisor == 0:
        return array
    return dividend / divisor


def get_mask_name(image_name):
    """"""
    image_name_array = image_name.split(SPLIT_CHARACTER)
    mask_name = SPLIT_CHARACTER.join(
        [MASK_BASE_NAME, image_name_array[-2], image_name_array[-1]]
    )
    return mask_name


class SARDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self._image_dir = image_dir
        self._mask_dir = mask_dir
        self._transform = transform
        self._images = os.listdir(image_dir)

    def __len__(self):
        return len(self._images)

    def __getitem__(self, index):
        img_path = os.path.join(self._image_dir, self._images[index])
        mask_name = get_mask_name(self._images[index])
        mask_path = os.path.join(self._mask_dir, mask_name)
        image = np.array(Image.open(img_path))
        mask = np.array(Image.open(mask_path).convert(MASK_COLOR_MAP), dtype=np.float32)
        image = normalize_array(image)
        # urban areas should be white
        mask[mask == 255.0] = 1.0
        # where the satellite image is blank, the mask should be black
        mask[image == 0.0] = 0.0

        if self._transform is not None:
            augmentations = self._transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask
