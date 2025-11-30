
from torch import device
import os
import time
import csv
import glob
import numpy as np
from ultralytics import YOLO
from pathlib import Path


def compute_iou(boxA, boxB):
    # box format: [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

# Load YOLO11 Model

model_path = "best.pt"
image_dir  = "inference_subset/images"
label_dir  = "inference_subset/labels"

model = YOLO(model_path)

csv_file = "inference_metrics_yolo2.csv"
csv_cols = [
    "image",
    "inference_time_ms",
    "num_predictions",
    "num_labels",
    "matched_detections",
    "precision",
    "recall",
    "mean_iou",
    "mean_confidence",
    "max_confidence",
    "min_confidence",
]

f = open(csv_file, "w", newline="")
writer = csv.DictWriter(f, fieldnames=csv_cols)
writer.writeheader()



all_precisions = []
all_recalls = []
all_ious = []
all_times = []

image_paths = sorted(glob.glob(os.path.join(image_dir, "*.*")))

for img_path in image_paths:

    img_name = Path(img_path).stem
    label_file = os.path.join(label_dir, f"{img_name}.txt")
   
    # Load ground-truth labels   
    gt_boxes = []
    if os.path.exists(label_file):
        with open(label_file, "r") as lf:
            for line in lf:
                cls, x, y, w, h = map(float, line.split())

                # Convert YOLO
                gt_boxes.append([cls, x, y, w, h])

  
    # Run inference & record time    
    t0 = time.time()
    results = model(img_path, save= False)
    infer_ms = (time.time() - t0) * 1000
    all_times.append(infer_ms)

    res = results[0]

    # Predictions
    pred_boxes = res.boxes.xyxy.cpu().numpy()  # xyxy
    pred_cls   = res.boxes.cls.cpu().numpy()
    pred_conf  = res.boxes.conf.cpu().numpy()

    num_pred = len(pred_boxes)
    num_gt   = len(gt_boxes)
  
    # Convert ground truth to xyxy  
    converted_gt = []
    if num_gt > 0:
        h, w = res.orig_shape
        for (cls, x, y, bw, bh) in gt_boxes:
            x1 = (x - bw/2) * w
            y1 = (y - bh/2) * h
            x2 = (x + bw/2) * w
            y2 = (y + bh/2) * h
            converted_gt.append([x1, y1, x2, y2, cls])

    
    # Match predictions to GT    
    matched = 0
    ious = []
    used_gt = set()

    for i in range(num_pred):
        box_pred = pred_boxes[i]
        best_iou = 0
        best_gt = None

        for j, gt in enumerate(converted_gt):
            if j in used_gt:
                continue
            iou = compute_iou(box_pred, gt[:4])
            if iou > best_iou:
                best_iou = iou
                best_gt = j

        if best_iou >= 0.5:  # IoU threshold
            matched += 1
            ious.append(best_iou)
            used_gt.add(best_gt)

    precision = matched / num_pred if num_pred > 0 else 0
    recall    = matched / num_gt if num_gt > 0 else 0
    mean_iou  = np.mean(ious) if ious else 0

    all_precisions.append(precision)
    all_recalls.append(recall)
    all_ious.append(mean_iou)

    
    # Write per-image metrics
    writer.writerow({
        "image": img_name,
        "inference_time_ms": infer_ms,
        "num_predictions": num_pred,
        "num_labels": num_gt,
        "matched_detections": matched,
        "precision": precision,
        "recall": recall,
        "mean_iou": mean_iou,
        "mean_confidence": float(np.mean(pred_conf)) if num_pred > 0 else 0,
        "max_confidence": float(np.max(pred_conf)) if num_pred > 0 else 0,
        "min_confidence": float(np.min(pred_conf)) if num_pred > 0 else 0,
    })

f.close()




print("\n====== Evaluation Summary ======")
print(f"Images evaluated: {len(image_paths)}")
print(f"CSV saved to:      {csv_file}")



