import os
import random

# 재현 가능성
random.seed(42)

# 이미지 폴더
image_dir = "data/raw/images"

# split 저장 폴더
split_dir = "splits"
os.makedirs(split_dir, exist_ok=True)

# 비율 설정
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# 이미지 파일 목록
image_files = [f for f in os.listdir(image_dir) if f.endswith(".png")]
image_files.sort()

print("전체 이미지 개수:", len(image_files))

# 섞기
random.shuffle(image_files)

# 개수 계산
total = len(image_files)
train_count = int(total * train_ratio)
val_count = int(total * val_ratio)
test_count = total - train_count - val_count

# 분할
train_files = image_files[:train_count]
val_files = image_files[train_count:train_count + val_count]
test_files = image_files[train_count + val_count:]

# 저장 함수
def save_list(file_list, path):
    with open(path, "w", encoding="utf-8") as f:
        for file_name in file_list:
            f.write(file_name + "\n")

# txt 저장
save_list(train_files, os.path.join(split_dir, "train.txt"))
save_list(val_files, os.path.join(split_dir, "val.txt"))
save_list(test_files, os.path.join(split_dir, "test.txt"))

# 출력
print("\n=== 데이터 분할 완료 ===")
print("Train:", len(train_files))
print("Valid:", len(val_files))
print("Test :", len(test_files))
print("총합  :", len(train_files) + len(val_files) + len(test_files))