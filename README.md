# Self-Driving Car Object Detection

## Project Overview

This project focuses on developing and evaluating object detection models for autonomous vehicles. We will fine-tune two state-of-the-art models—YOLOv11 and RF-DETR—and compare their performance on a self-driving car dataset.

## Objectives

- Fine-tune YOLOv11 and RF-DETR models for object detection tasks relevant to autonomous driving.
- Compare the capabilities and performance of both models.
- Analyze results and provide insights for future improvements.

## Dataset

We use the [Self-Driving Car Object Detection Dataset](https://public.roboflow.com/object-detection/self-driving-car/3) from Roboflow, which contains annotated images of various objects encountered by autonomous vehicles.

## Models

- **YOLOv11**: A fast, real-time object detection model.
- **RF-DETR**: A transformer-based detection model with robust feature extraction.

## Workflow

1. **Data Preparation**: Download and preprocess the dataset.
2. **Model Fine-Tuning**: Train YOLOv11 and RF-DETR on the dataset.
3. **Evaluation**: Assess model performance using standard metrics.
4. **Comparison & Analysis**: Compare results and discuss findings.

## Getting Started

1. Clone the repository:
2. Install UV for dependency management:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

3. Install dependencies:

```bash
uv sync
```

## Results

Results and analysis will be added after model training and evaluation.
