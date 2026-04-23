import os

image_dir = "data/raw/images"
mask_dir = "data/processed/masks"

images = [f for f in os.listdir(image_dir) if f.endswith('.png')]
masks = [f for f in os.listdir(mask_dir) if f.endswith('.png')]

print("=== 데이터 개수 ===")
print("이미지:", len(images))
print("마스크:", len(masks))


def count_txt(file):
    with open(file, 'r') as f:
        return len(f.readlines())

print("\n=== 데이터 분할 ===")
print("Train:", count_txt("splits/train.txt"))
print("Valid:", count_txt("splits/val.txt"))
print("Test :", count_txt("splits/test.txt"))