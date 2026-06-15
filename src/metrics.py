import numpy as np

def compute_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    intersection = inter_w * inter_h

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    if union <= 0.0:
        return 0.0
    return intersection / union

def evaluate_detections(pred_boxes, pred_labels, gt_boxes, gt_labels, iou_thresh=0.25):
    if len(gt_boxes) == 0:
        if len(pred_boxes) == 0:
            return 1.0, 1.0, 1.0
        return 0.0, 0.0, 0.0

    if len(pred_boxes) == 0:
        return 0.0, 0.0, 0.0

    tp = 0
    fp = 0
    matched_gt = set()

    for p_idx, p_box in enumerate(pred_boxes):
        best_iou = 0.0
        best_gt_idx = -1

        for g_idx, g_box in enumerate(gt_boxes):
            # Enforce 1-to-1 matching: once matched, GT box cannot be claimed by another detection
            if g_idx in matched_gt:
                continue

            iou = compute_iou(p_box, g_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = g_idx

        if best_iou >= iou_thresh and best_gt_idx != -1:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1