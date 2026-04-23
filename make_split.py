import os
import random

random.seed(42)

image_dir = "data/raw/images"

split_dir = "splits"
os.makedirs(split_dir, exist_ok=True)

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

image_files = [f for f in os.listdir(image_dir) if f.endswith(".png")]
image_files.sort()

print("전체 이미지 개수:", len(image_files))

random.shuffle(image_files)

total = len(image_files)
train_count = int(total * train_ratio)
val_count = int(total * val_ratio)
test_count = total - train_count - val_count

train_files = image_files[:train_count]
val_files = image_files[train_count:train_count + val_count]
test_files = image_files[train_count + val_count:]

def save_list(file_list, path):
    with open(path, "w", encoding="utf-8") as f:
        for file_name in file_list:
            f.write(file_name + "\n")

save_list(train_files, os.path.join(split_dir, "train.txt"))
save_list(val_files, os.path.join(split_dir, "val.txt"))
save_list(test_files, os.path.join(split_dir, "test.txt"))

print("\n=== 데이터 분할 완료 ===")
print("Train:", len(train_files))
print("Valid:", len(val_files))
print("Test :", len(test_files))
print("총합  :", len(train_files) + len(val_files) + len(test_files))
