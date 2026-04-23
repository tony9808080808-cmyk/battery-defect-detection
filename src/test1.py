import cv2
import numpy as np
import os

mask_dir = "data/processed/masks"

bg, dam, pol = 0, 0, 0

for f in os.listdir(mask_dir):
    m = cv2.imread(os.path.join(mask_dir, f), 0)

    bg += int(np.sum(m == 0))
    dam += int(np.sum(m == 127))
    pol += int(np.sum(m == 255))

print(bg, dam, pol)