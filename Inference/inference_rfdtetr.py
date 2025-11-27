from rfdetr import RFDETRNano
from PIL import Image
import supervision as sv
from tqdm import tqdm
import time
import csv
import os
import numpy as np


MODEL_WEIGHTS = "checkpoint_best_total.pth"
IMAGES_DIR = "inference_subset/images"
ANNOTATIONS = "inference_subset/filtered_annotations.coco.json"
CSV_PATH = "inference_metrics_rfdetr.csv"
SCORE_THR = 0.001
IOU_MATCH_THR = 0.5


model = RFDETRNano(pretrain_weights=MODEL_WEIGHTS)


val_ds = sv.DetectionDataset.from_coco(
    images_directory_path=IMAGES_DIR,
    annotations_path=ANNOTATIONS
)


with open(CSV_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "image",
        "inference_time_ms",
        "num_predictions",
        "num_labels",
        "matched_detections",
        "precision",
        "recall",
        "mean_iou"
    ])
    writer.writeheader()


# IOU FUNCTION
def compute_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    area1 = max(0, (box1[2] - box1[0])) * max(0, (box1[3] - box1[1]))
    area2 = max(0, (box2[2] - box2[0])) * max(0, (box2[3] - box2[1]))
    union = area1 + area2 - inter + 1e-6

    return inter / union if union > 0 else 0.0


# PROCESS IMAGES
for path, image, gt_ann in tqdm(val_ds):
    img_name = os.path.basename(path)

    # Handle image type
    if isinstance(image, np.ndarray):
        image_pil = Image.fromarray(image)
    else:
        image_pil = image

    
    # RUN INFERENCE
    
    t0 = time.time()
    detections = model.predict(image_pil, threshold=0.5)
    infer_ms = (time.time() - t0) * 1000

 
    # FILTER PREDICTIONS
    
    valid_dets = [det for det in detections if det[0] is not None and det[2] >= SCORE_THR]
    pred_boxes = [det[0] for det in valid_dets]
    pred_scores = [float(det[2]) for det in valid_dets]
    pred_classes = [int(det[3]) for det in valid_dets]

    num_pred = len(pred_boxes)

    
    # PROCESS GROUND TRUTH
    
    gt_boxes = []
    gt_classes = []

    for ann in gt_ann:
        box = ann[0]
        cls = ann[3]
        if box is None:
            continue
        x1, y1, x2, y2 = [float(x) for x in box]
        if x2 <= x1 or y2 <= y1:
            continue
        gt_boxes.append([x1, y1, x2, y2])
        gt_classes.append(int(cls))

    num_gt = len(gt_boxes)

 
    # MATCH PRED  
    matched = 0
    ious = []
    used_gt = set()

    for pb in pred_boxes:
        best_iou = 0.0
        best_gt_idx = -1
        for i, gb in enumerate(gt_boxes):
            if i in used_gt:
                continue
            iou = compute_iou(pb, gb)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        if best_iou >= IOU_MATCH_THR:
            matched += 1
            used_gt.add(best_gt_idx)
            ious.append(best_iou)

    precision = matched / num_pred if num_pred > 0 else 0.0
    recall = matched / num_gt if num_gt > 0 else 0.0
    mean_iou = float(np.mean(ious)) if len(ious) > 0 else 0.0

    # WRITE METRICS
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image","inference_time_ms","num_predictions","num_labels",
            "matched_detections","precision","recall","mean_iou"
        ])
        writer.writerow({
            "image": img_name,
            "inference_time_ms": round(infer_ms, 2),
            "num_predictions": num_pred,
            "num_labels": num_gt,
            "matched_detections": matched,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "mean_iou": round(mean_iou, 4)
        })

print("Inference complete.")
print("CSV saved to:", CSV_PATH)
