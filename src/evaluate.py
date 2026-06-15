import os
import torch
import pandas as pd
from torchvision.ops import nms
from src.metrics import evaluate_detections
from src.visualize import draw_crop_detections

def run_evaluation(model, dataloader, device="cpu", output_csv="outputs/results_summary.csv"):
    model.eval()
    model.to(device)

    results = []

    with torch.no_grad():
        for images, targets, filenames in dataloader:
            images = [img.to(device) for img in images]
            predictions = model(images)

            for idx, pred in enumerate(predictions):
                img_name = filenames[idx]
                gt_boxes = targets[idx]["boxes"].cpu().numpy().tolist()
                gt_labels = targets[idx]["labels"].cpu().numpy().tolist()

                raw_boxes = pred["boxes"]
                raw_scores = pred["scores"]
                raw_labels = pred["labels"]
                

                score_thresh = 0.45
                keep_mask = raw_scores >= score_thresh

                filt_boxes = raw_boxes[keep_mask]
                filt_scores = raw_scores[keep_mask]
                filt_labels = raw_labels[keep_mask]

                if len(filt_boxes) > 0:

                    keep_indices = nms(filt_boxes, filt_scores, iou_threshold=0.30)
                    pred_boxes = filt_boxes[keep_indices].cpu().numpy().tolist()
                    pred_scores = filt_scores[keep_indices].cpu().numpy().tolist()
                    pred_labels = filt_labels[keep_indices].cpu().numpy().tolist()
                else:
                    pred_boxes, pred_scores, pred_labels = [], [], []

                precision, recall, f1 = evaluate_detections(
                    pred_boxes, pred_labels, gt_boxes, gt_labels, iou_thresh=0.25
                )

                results.append({
                    "Image": img_name,
                    "Ground_Truth_Count": len(gt_boxes),
                    "Detected_Count": len(pred_boxes),
                    "Precision": round(precision, 4),
                    "Recall": round(recall, 4),
                    "F1_Score": round(f1, 4)
                })

                img_path = os.path.join(dataloader.dataset.ImgDir, img_name)
                out_path = os.path.join("outputs/figures", f"pred_{img_name}")
                draw_crop_detections(img_path, pred_boxes, pred_labels, pred_scores, out_path, threshold=0.45)

    summary_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    summary_df.to_csv(output_csv, index=False)

    print("\n================ EVALUATION SUMMARY ================")
    print(summary_df.to_string(index=False))
    print("====================================================")
    return summary_df