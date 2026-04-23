import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class BatteryDataset(Dataset):
    def __init__(self, image_dir, mask_dir, file_list, image_size=(128, 128)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.file_list = file_list
        self.image_size = image_size

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]

        file_name = os.path.basename(file_name)

        image_path = os.path.join(self.image_dir, file_name)
        mask_path = os.path.join(self.mask_dir, file_name)

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"마스크를 찾을 수 없습니다: {mask_path}")

        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)

        unique_values = np.unique(mask)
        valid_values = {0, 127, 255}
        if not set(unique_values).issubset(valid_values):
            raise ValueError(
                f"마스크에 예상하지 못한 값이 있습니다: {mask_path}, unique={unique_values}"
            )

        mask_converted = np.zeros_like(mask, dtype=np.uint8)
        mask_converted[mask == 127] = 1
        mask_converted[mask == 255] = 2

        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))

        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask_converted, dtype=torch.long)

        return image, mask


def load_file_list(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        file_list = [line.strip() for line in f if line.strip()]
    return file_list
