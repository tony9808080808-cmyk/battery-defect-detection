import os
import random

image_dir = "data/raw/images"

output_txt = "splits/test.txt"

num_samples = 200

all_files = os.listdir(image_dir)

image_files = [f for f in all_files if f.endswith(".png")]

print("전체 이미지 개수:", len(image_files))

selected_files = random.sample(image_files, num_samples)

with open(output_txt, "w", encoding="utf-8") as f:
    for file_name in selected_files:
        f.write(file_name + "\n")

print(f"{num_samples}개 test 데이터 생성 완료 → {output_txt}")
