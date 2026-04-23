import os
import random

# 이미지 폴더 경로
image_dir = "data/raw/images"

# 저장할 파일 경로
output_txt = "splits/test.txt"

# 몇 개 뽑을지
num_samples = 200

# 이미지 파일 리스트 가져오기
all_files = os.listdir(image_dir)

# 이미지 파일만 필터링 (png 기준)
image_files = [f for f in all_files if f.endswith(".png")]

print("전체 이미지 개수:", len(image_files))

# 랜덤으로 200개 선택
selected_files = random.sample(image_files, num_samples)

# txt 파일로 저장
with open(output_txt, "w", encoding="utf-8") as f:
    for file_name in selected_files:
        f.write(file_name + "\n")

print(f"{num_samples}개 test 데이터 생성 완료 → {output_txt}")