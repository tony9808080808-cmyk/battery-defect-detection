import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import BatteryDataset, load_file_list
from model import UNet
from visualize import save_colored_prediction


def calculate_iou_dice(pred_mask, true_mask, num_classes=3):
    iou_list = []
    dice_list = []

    for cls in range(num_classes):
        pred_cls = (pred_mask == cls)
        true_cls = (true_mask == cls)

        intersection = np.logical_and(pred_cls, true_cls).sum()
        union = np.logical_or(pred_cls, true_cls).sum()

        pred_sum = pred_cls.sum()
        true_sum = true_cls.sum()

        if union == 0:
            iou = np.nan
        else:
            iou = intersection / union

        if pred_sum + true_sum == 0:
            dice = np.nan
        else:
            dice = (2 * intersection) / (pred_sum + true_sum)

        iou_list.append(iou)
        dice_list.append(dice)

    return iou_list, dice_list


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("사용 디바이스:", device)

    # =========================
    # 1. 경로 설정
    # =========================
    model_path = "checkpoints/best_weighted_model_epoch9.pth"

    image_dir = "data/raw/images"
    mask_dir = "data/processed/masks"
    test_txt = "splits/test.txt"

    save_pred_dir = "results_test/predictions"
    save_vis_dir = "results_test/visualization"
    os.makedirs(save_pred_dir, exist_ok=True)
    os.makedirs(save_vis_dir, exist_ok=True)

    # =========================
    # 2. 테스트 파일 로드
    # =========================
    test_files = load_file_list(test_txt)

    test_dataset = BatteryDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        file_list=test_files,
        image_size=(128, 128)
    )

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # =========================
    # 3. 모델 로드
    # =========================
    model = UNet(in_channels=3, out_channels=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    class_names = ["Background", "Damaged", "Pollution"]

    all_ious = [[] for _ in range(3)]
    all_dices = [[] for _ in range(3)]

    # =========================
    # 4. 추론 시작
    # =========================
    with torch.no_grad():
        for idx, (images, masks) in enumerate(tqdm(test_loader, desc="[Test Inference]")):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            image_np = images[0].cpu().numpy().transpose(1, 2, 0)   # (H, W, C), float
            true_mask_np = masks[0].cpu().numpy().astype(np.uint8)  # (H, W)
            pred_mask_np = preds[0].cpu().numpy().astype(np.uint8)  # (H, W)

            # -------------------------
            # 성능 계산
            # -------------------------
            iou_list, dice_list = calculate_iou_dice(pred_mask_np, true_mask_np, num_classes=3)

            for cls in range(3):
                if not np.isnan(iou_list[cls]):
                    all_ious[cls].append(iou_list[cls])
                if not np.isnan(dice_list[cls]):
                    all_dices[cls].append(dice_list[cls])

            # -------------------------
            # 파일명 / 저장 경로
            # -------------------------
            file_name = os.path.basename(test_files[idx])
            pred_save_path = os.path.join(save_pred_dir, file_name)

            # -------------------------
            # prediction 저장
            # predictions 폴더에는 0/127/255 형태로 저장
            # -------------------------
            pred_save_mask = pred_mask_np.copy()
            pred_save_mask[pred_save_mask == 1] = 127
            pred_save_mask[pred_save_mask == 2] = 255
            cv2.imwrite(pred_save_path, pred_save_mask)

            # -------------------------
            # visualization 저장
            # -------------------------
            input_path = os.path.join(image_dir, file_name)
            gt_path = os.path.join(mask_dir, file_name)

            save_colored_prediction(
                input_path=input_path,
                gt_path=gt_path,
                pred_path=pred_save_path,
                save_dir=save_vis_dir
            )

    # =========================
    # 5. 성능 출력
    # =========================
    print("\n===== 클래스별 평균 성능 =====")
    for cls in range(3):
        mean_iou = np.mean(all_ious[cls]) if len(all_ious[cls]) > 0 else 0.0
        mean_dice = np.mean(all_dices[cls]) if len(all_dices[cls]) > 0 else 0.0

        print(f"{class_names[cls]}")
        print(f"  Mean IoU : {mean_iou:.4f}")
        print(f"  Mean Dice: {mean_dice:.4f}")

    valid_iou_means = [
        np.mean(all_ious[cls]) for cls in range(3) if len(all_ious[cls]) > 0
    ]
    valid_dice_means = [
        np.mean(all_dices[cls]) for cls in range(3) if len(all_dices[cls]) > 0
    ]

    overall_mean_iou = np.mean(valid_iou_means) if len(valid_iou_means) > 0 else 0.0
    overall_mean_dice = np.mean(valid_dice_means) if len(valid_dice_means) > 0 else 0.0

    print("\n===== 전체 평균 성능 =====")
    print(f"Mean IoU : {overall_mean_iou:.4f}")
    print(f"Mean Dice: {overall_mean_dice:.4f}")

    # =========================
    # 6. metrics 저장
    # =========================
    metrics_path = "results_test/metrics.txt"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("===== 클래스별 평균 성능 =====\n")
        for cls in range(3):
            mean_iou = np.mean(all_ious[cls]) if len(all_ious[cls]) > 0 else 0.0
            mean_dice = np.mean(all_dices[cls]) if len(all_dices[cls]) > 0 else 0.0

            f.write(f"{class_names[cls]}\n")
            f.write(f"  Mean IoU : {mean_iou:.4f}\n")
            f.write(f"  Mean Dice: {mean_dice:.4f}\n")

        f.write("\n===== 전체 평균 성능 =====\n")
        f.write(f"Mean IoU : {overall_mean_iou:.4f}\n")
        f.write(f"Mean Dice: {overall_mean_dice:.4f}\n")

    print(f"\n결과 저장 완료: {metrics_path}")
    print(f"예측 마스크 저장 폴더: {save_pred_dir}")
    print(f"시각화 저장 폴더: {save_vis_dir}")


if __name__ == "__main__":
    main()