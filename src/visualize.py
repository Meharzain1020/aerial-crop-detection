import os
import cv2

def draw_crop_detections(img_path, boxes, labels, scores, output_path, threshold=0.25):
    img = cv2.imread(img_path)
    if img is None:
        return

    colors = {1: (0, 255, 0), 2: (0, 165, 255)}
    names = {1: "Crop_Healthy", 2: "Crop_Stressed"}

    for box, label, score in zip(boxes, labels, scores):
        if score < threshold:
            continue
        xmin, ymin, xmax, ymax = map(int, box)
        color = colors.get(label, (255, 0, 0))
        name = names.get(label, "Crop")

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(img, f"{name} {score:.2f}", (xmin, max(15, ymin - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)