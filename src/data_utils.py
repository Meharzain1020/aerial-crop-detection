import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

def generate_synthetic_crop_data(output_dir, num_samples=10):
    os.makedirs(output_dir, exist_ok=True)
    img_dir = os.path.join(output_dir, "images")
    lbl_dir = os.path.join(output_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    for idx in range(num_samples):
        img = np.random.randint(100, 180, (640, 640, 3), dtype=np.uint8)
        boxes = []
        num_objects = np.random.randint(2, 5)

        for _ in range(num_objects):
            xmin = float(np.random.randint(40, 420))
            ymin = float(np.random.randint(40, 420))
            xmax = float(xmin + np.random.randint(60, 150))
            ymax = float(ymin + np.random.randint(60, 150))
            label = np.random.randint(1, 3)

            cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (34, 139, 34), -1)
            boxes.append(f"{label} {xmin} {ymin} {xmax} {ymax}")

        img_path = os.path.join(img_dir, f"crop_{idx:03d}.jpg")
        lbl_path = os.path.join(lbl_dir, f"crop_{idx:03d}.txt")
        cv2.imwrite(img_path, img)

        with open(lbl_path, "w") as f:
            f.write("\n".join(boxes))

class AerialCropDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.ImgDir = os.path.join(data_dir, "images")
        self.LblDir = os.path.join(data_dir, "labels")
        self.ImageFiles = sorted([f for f in os.listdir(self.ImgDir) if f.endswith(('.jpg', '.png'))])

    def __len__(self):
        return len(self.ImageFiles)

    def __getitem__(self, idx):
        filename = self.ImageFiles[idx]
        img_path = os.path.join(self.ImgDir, filename)
        lbl_path = os.path.join(self.LblDir, filename.replace(os.path.splitext(filename)[1], '.txt'))

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        boxes_list = []
        labels_list = []

        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        labels_list.append(int(parts[0]))
                        boxes_list.append([float(x) for x in parts[1:]])

        boxes = torch.as_tensor(boxes_list, dtype=torch.float32) if len(boxes_list) > 0 else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.as_tensor(labels_list, dtype=torch.int64) if len(labels_list) > 0 else torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }

        img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        return img_tensor, target, filename