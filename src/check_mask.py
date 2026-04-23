import cv2
import matplotlib.pyplot as plt
from pathlib import Path

image_dir = Path("data/raw/images")
mask_dir = Path("data/processed/masks")

# 아무 이미지 하나 선택
image_path = list(image_dir.glob("*.png"))[0]
mask_path = mask_dir / image_path.name

# 읽기
image = cv2.imread(str(image_path))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

# 시각화
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.title("Image")
plt.imshow(image)

plt.subplot(1,2,2)
plt.title("Mask")
plt.imshow(mask, cmap='gray')

plt.show()