import os
import cv2
import numpy as np


def _to_class_index(mask: np.ndarray) -> np.ndarray:
    mask = mask.copy()
    unique_vals = np.unique(mask)

    if set(unique_vals).issubset({0, 1, 2}):
        return mask

    mask[mask == 127] = 1
    mask[mask == 255] = 2
    return mask


def _make_color_mask(mask: np.ndarray) -> np.ndarray:
    color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    color[mask == 1] = [0, 0, 255]      # Damaged -> Red
    color[mask == 2] = [0, 255, 255]    # Pollution -> Yellow

    return color


def save_colored_prediction(
    input_path: str,
    gt_path: str,
    pred_path: str,
    save_dir: str = "results/visualization"
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    image = cv2.imread(input_path)
    gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"입력 이미지 로드 실패: {input_path}")
    if gt_mask is None:
        raise FileNotFoundError(f"GT 마스크 로드 실패: {gt_path}")
    if pred_mask is None:
        raise FileNotFoundError(f"예측 마스크 로드 실패: {pred_path}")

    h, w = image.shape[:2]
    if gt_mask.shape[:2] != (h, w):
        gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    if pred_mask.shape[:2] != (h, w):
        pred_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    gt_mask = _to_class_index(gt_mask)
    pred_mask = _to_class_index(pred_mask)

    gt_color = _make_color_mask(gt_mask)
    pred_color = _make_color_mask(pred_mask)

    gt_overlay = cv2.addWeighted(image, 0.7, gt_color, 0.3, 0)
    pred_overlay = cv2.addWeighted(image, 0.7, pred_color, 0.3, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    text_color = (255, 255, 255)

    def add_label(img: np.ndarray, text: str) -> np.ndarray:
        labeled = img.copy()
        cv2.putText(labeled, text, (15, 30), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        return labeled

    input_labeled = add_label(image, "Input")
    gt_labeled = add_label(gt_overlay, "Ground Truth")
    pred_labeled = add_label(pred_overlay, "Prediction")

    triplet = np.hstack([input_labeled, gt_labeled, pred_labeled])

    base_name = os.path.splitext(os.path.basename(input_path))[0]

    cv2.imwrite(os.path.join(save_dir, f"{base_name}_input.png"), image)
    cv2.imwrite(os.path.join(save_dir, f"{base_name}_gt_color.png"), gt_color)
    cv2.imwrite(os.path.join(save_dir, f"{base_name}_pred_color.png"), pred_color)
    cv2.imwrite(os.path.join(save_dir, f"{base_name}_gt_overlay.png"), gt_overlay)
    cv2.imwrite(os.path.join(save_dir, f"{base_name}_pred_overlay.png"), pred_overlay)
    cv2.imwrite(os.path.join(save_dir, f"{base_name}_triplet.png"), triplet)

    print(f"[완료] 시각화 저장: {base_name}")