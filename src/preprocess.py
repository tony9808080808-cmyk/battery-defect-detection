import json
from pathlib import Path

import cv2
import numpy as np


# 클래스 매핑
# 시각화 확인을 위해 사람이 구분 가능한 값으로 저장
CLASS_MAP = {
    "Damaged": 127,   # 회색
    "Pollution": 255  # 흰색
}


def load_json(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_matching_image(image_dir: Path, stem: str):
    image_extensions = [".png", ".jpg", ".jpeg", ".bmp"]

    for ext in image_extensions:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate

    return None


def extract_defects(data: dict):
    if "defects" in data and isinstance(data["defects"], list):
        return data["defects"]
    return []


def points_to_polygon(points, json_path: Path):
    if not isinstance(points, list):
        print(f"[경고] points 형식 오류: {json_path.name}")
        return None

    if len(points) < 6 or len(points) % 2 != 0:
        print(f"[경고] polygon 점 개수 이상: {json_path.name}")
        return None

    polygon = np.array(points, dtype=np.float32).reshape(-1, 2)
    polygon = np.round(polygon).astype(np.int32)

    return polygon


def clip_polygon_to_image(polygon: np.ndarray, width: int, height: int):
    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
    return polygon


def json_to_mask(json_path: Path, image_path: Path, save_path: Path):
    data = load_json(json_path)

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {image_path}")

    height, width = image.shape[:2]

    mask = np.zeros((height, width), dtype=np.uint8)

    defects = extract_defects(data)

    if len(defects) == 0:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), mask)
        return {
            "status": "empty",
            "defect_count": 0,
            "class_pixels": {"Damaged": 0, "Pollution": 0}
        }

    defect_count = 0
    class_pixels = {"Damaged": 0, "Pollution": 0}

    for defect in defects:
        label_name = defect.get("name")
        points = defect.get("points", [])

        if label_name not in CLASS_MAP:
            print(f"[경고] 알 수 없는 클래스: {label_name} | 파일: {json_path.name}")
            continue

        polygon = points_to_polygon(points, json_path)
        if polygon is None:
            continue

        polygon = clip_polygon_to_image(polygon, width, height)
        class_value = CLASS_MAP[label_name]

        cv2.fillPoly(mask, [polygon], class_value)
        defect_count += 1

    class_pixels["Damaged"] = int((mask == 127).sum())
    class_pixels["Pollution"] = int((mask == 255).sum())

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), mask)

    return {
        "status": "ok",
        "defect_count": defect_count,
        "class_pixels": class_pixels
    }


def process_all():
    image_dir = Path("data/raw/images")
    label_dir = Path("data/raw/labels")
    mask_dir = Path("data/processed/masks")

    if not image_dir.exists():
        print(f"[에러] 이미지 폴더가 없습니다: {image_dir}")
        return

    if not label_dir.exists():
        print(f"[에러] 라벨 폴더가 없습니다: {label_dir}")
        return

    json_files = sorted(label_dir.glob("*.json"))

    if not json_files:
        print("[에러] JSON 파일이 없습니다.")
        return

    total_json = len(json_files)
    success_count = 0
    empty_count = 0
    skipped_count = 0
    error_count = 0

    total_defects = 0
    total_class_pixels = {"Damaged": 0, "Pollution": 0}

    print("=" * 60)
    print("마스크 생성 시작")
    print(f"라벨 파일 수: {total_json}")
    print("=" * 60)

    for idx, json_file in enumerate(json_files, start=1):
        stem = json_file.stem
        image_file = find_matching_image(image_dir, stem)

        if image_file is None:
            print(f"[건너뜀] 매칭되는 이미지 없음: {stem}")
            skipped_count += 1
            continue

        save_file = mask_dir / f"{stem}.png"

        try:
            result = json_to_mask(json_file, image_file, save_file)

            if result["status"] == "empty":
                empty_count += 1
            else:
                success_count += 1
                total_defects += result["defect_count"]

                total_class_pixels["Damaged"] += result["class_pixels"]["Damaged"]
                total_class_pixels["Pollution"] += result["class_pixels"]["Pollution"]

            if idx % 100 == 0 or idx == total_json:
                print(f"[진행] {idx}/{total_json} 처리 완료")

        except Exception as e:
            print(f"[에러] {stem} -> {e}")
            error_count += 1

    print("=" * 60)
    print("마스크 생성 완료")
    print(f"총 JSON 수: {total_json}")
    print(f"성공: {success_count}")
    print(f"결함 없음(빈 mask): {empty_count}")
    print(f"건너뜀(매칭 이미지 없음): {skipped_count}")
    print(f"에러: {error_count}")
    print(f"총 defect 수: {total_defects}")
    print(f"Damaged 총 픽셀 수: {total_class_pixels['Damaged']}")
    print(f"Pollution 총 픽셀 수: {total_class_pixels['Pollution']}")
    print("=" * 60)


if __name__ == "__main__":
    process_all()
