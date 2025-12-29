#!/usr/bin/env python3
"""
YOLOv8-AM Fracture Detection Script with Comprehensive Metrics
================================================================

This script performs inference using YOLOv8 models with attention mechanisms
and calculates detailed evaluation metrics including precision, recall, F1-score,
mAP, and accuracy.

Compatible with models: YOLOv8, YOLOv8+SA, YOLOv8+ECA, YOLOv8+GAM, YOLOv8+ResCBAM

Author: Detection Script for YOLOv8-AM
Usage:
    python detect.py --model path/to/model.pt --source path/to/images [options]
"""

import argparse
import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import json
import warnings
from collections import defaultdict

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Fix numpy compatibility issues with older libraries
if not hasattr(np, 'object'):
    np.object = object
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float

# Add ultralytics to path if it exists
if os.path.exists('./ultralytics'):
    sys.path.insert(0, './ultralytics')

try:
    from ultralytics import YOLO
    from ultralytics.utils.plotting import Annotator, colors
except ImportError:
    print("Error: ultralytics not found. Please install or check path.")
    print("pip install ultralytics")
    sys.exit(1)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='YOLOv8-AM Fracture Detection with Metrics',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained YOLOv8 model weights (.pt file)')
    parser.add_argument('--source', type=str, required=True,
                       help='Path to test images directory or single image')
    
    # Model parameters
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Input image size')
    parser.add_argument('--conf', type=float, default=0.20,
                       help='Confidence threshold for predictions')
    parser.add_argument('--iou', type=float, default=0.30,
                       help='IoU threshold for NMS')
    parser.add_argument('--device', type=str, default='',
                       help='Device to run inference on (0, cpu, cuda:0, etc.)')
    
    # Metrics parameters
    parser.add_argument('--labels-dir', type=str, default=None,
                       help='Path to ground truth labels directory (YOLO format)')
    parser.add_argument('--iou-thres-metrics', type=float, default=0.5,
                       help='IoU threshold for metrics calculation')
    parser.add_argument('--save-metrics', action='store_true',
                       help='Calculate and save detailed metrics (requires --labels-dir)')
    
    # Output parameters
    parser.add_argument('--save-img', action='store_true', 
                       help='Save annotated images')
    parser.add_argument('--save-txt', action='store_true',
                       help='Save detection results in YOLO format')
    parser.add_argument('--save-json', action='store_true',
                       help='Save results in JSON format')
    parser.add_argument('--project', type=str, default='runs/detect',
                       help='Save results to project directory')
    parser.add_argument('--name', type=str, default='exp',
                       help='Save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                       help='Allow existing project/name, do not increment')
    
    # Visualization parameters
    parser.add_argument('--line-thickness', type=int, default=3,
                       help='Bounding box line thickness')
    parser.add_argument('--hide-labels', action='store_true',
                       help='Hide class labels in saved images')
    parser.add_argument('--hide-conf', action='store_true',
                       help='Hide confidence scores in saved images')
    
    # Advanced parameters
    parser.add_argument('--classes', nargs='+', type=int,
                       help='Filter by specific classes')
    parser.add_argument('--agnostic-nms', action='store_true',
                       help='Use class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                       help='Use augmented inference')
    
    return parser.parse_args()


def setup_device(device_arg):
    """Setup and validate device configuration"""
    if device_arg:
        if device_arg.lower() == 'cpu':
            return 'cpu'
        elif device_arg.isdigit():
            device_num = int(device_arg)
            if torch.cuda.is_available() and device_num < torch.cuda.device_count():
                return f'cuda:{device_num}'
            else:
                print(f"Warning: CUDA device {device_num} not available. Using CPU.")
                return 'cpu'
        elif device_arg.startswith('cuda'):
            if torch.cuda.is_available():
                return device_arg
            else:
                print("Warning: CUDA not available. Using CPU.")
                return 'cpu'
        else:
            return device_arg
    else:
        # Auto-detect
        return 'cuda:0' if torch.cuda.is_available() else 'cpu'


def load_model(model_path, device):
    """Load YOLOv8 model with proper error handling"""
    try:
        print(f"Loading model: {model_path}")
        print(f"Using device: {device}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = YOLO(model_path)
        model.to(device)
        
        print(f"Model loaded successfully!")
        print(f"Model classes: {model.names}")
        print(f"Number of classes: {len(model.names)}")
        
        return model
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        if torch.cuda.is_available():
            print(f"Available CUDA devices: {torch.cuda.device_count()}")
        return None


def get_image_files(source_path):
    """Get list of valid image files from source"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    if os.path.isfile(source_path):
        if Path(source_path).suffix.lower() in image_extensions:
            return [source_path]
        else:
            print(f"Error: {source_path} is not a valid image file")
            return []
    
    elif os.path.isdir(source_path):
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(source_path).glob(f'*{ext}'))
            image_files.extend(Path(source_path).glob(f'*{ext.upper()}'))
        
        return sorted([str(f) for f in image_files])
    
    else:
        print(f"Error: {source_path} does not exist")
        return []


def create_output_directory(project, name, exist_ok):
    """Create output directory with proper naming"""
    if exist_ok:
        save_dir = Path(project) / name
    else:
        save_dir = Path(project) / name
        counter = 1
        while save_dir.exists():
            save_dir = Path(project) / f"{name}{counter}"
            counter += 1
    
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def load_ground_truth_labels(labels_dir, image_files):
    """Load ground truth labels in YOLO format"""
    gt_data = {}
    
    if not labels_dir or not os.path.exists(labels_dir):
        print(f"Warning: Labels directory not found: {labels_dir}")
        return gt_data
    
    labels_found = 0
    for img_path in image_files:
        img_name = Path(img_path).stem
        label_path = os.path.join(labels_dir, f"{img_name}.txt")
        
        if os.path.exists(label_path):
            gt_boxes = []
            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split()
                            if len(parts) >= 5:
                                cls = int(parts[0])
                                x_center, y_center, width, height = map(float, parts[1:5])
                                gt_boxes.append([cls, x_center, y_center, width, height])
                gt_data[img_path] = gt_boxes
                if gt_boxes:
                    labels_found += 1
            except Exception as e:
                print(f"Error reading label file {label_path}: {e}")
                gt_data[img_path] = []
        else:
            gt_data[img_path] = []
    
    print(f"Loaded ground truth for {labels_found}/{len(image_files)} images")
    return gt_data


def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes (x1, y1, x2, y2 format)"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def yolo_to_xyxy(yolo_box, img_width, img_height):
    """Convert YOLO format to xyxy format"""
    x_center, y_center, width, height = yolo_box
    
    x1 = (x_center - width / 2) * img_width
    y1 = (y_center - height / 2) * img_height
    x2 = (x_center + width / 2) * img_width
    y2 = (y_center + height / 2) * img_height
    
    return [x1, y1, x2, y2]


def calculate_ap(precisions, recalls):
    """Calculate Average Precision using 11-point interpolation"""
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11
    return ap


def calculate_comprehensive_metrics(predictions, ground_truths, num_classes, iou_threshold=0.5):
    """Calculate comprehensive detection metrics"""
    
    # Initialize per-class metrics
    class_metrics = {}
    for cls in range(num_classes):
        class_metrics[cls] = {
            'tp': 0, 'fp': 0, 'fn': 0,
            'precisions': [], 'recalls': [], 'confidences': []
        }
    
    total_gt_boxes = 0
    total_pred_boxes = 0
    
    print("Calculating metrics for each image...")
    
    # Process each image
    for img_path, pred_results in tqdm(predictions.items(), desc="Processing metrics"):
        gt_boxes = ground_truths.get(img_path, [])
        
        # Get image dimensions
        if hasattr(pred_results, 'orig_shape') and pred_results.orig_shape is not None:
            img_height, img_width = pred_results.orig_shape
        else:
            # Try to get from image file
            try:
                img = cv2.imread(img_path)
                img_height, img_width = img.shape[:2]
            except:
                img_height, img_width = 1024, 1024  # Default
        
        # Convert ground truth to xyxy format
        gt_xyxy = []
        gt_classes = []
        for gt_box in gt_boxes:
            cls, x_center, y_center, width, height = gt_box
            xyxy = yolo_to_xyxy([x_center, y_center, width, height], img_width, img_height)
            gt_xyxy.append(xyxy)
            gt_classes.append(int(cls))
        
        total_gt_boxes += len(gt_boxes)
        
        # Get predictions
        pred_boxes = []
        pred_classes = []
        pred_confs = []
        
        if pred_results.boxes is not None and len(pred_results.boxes) > 0:
            for box in pred_results.boxes:
                cls = int(box.cls.cpu().numpy()[0])
                conf = float(box.conf.cpu().numpy()[0])
                x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
                
                pred_boxes.append([x1, y1, x2, y2])
                pred_classes.append(cls)
                pred_confs.append(conf)
        
        total_pred_boxes += len(pred_boxes)
        
        # Match predictions to ground truth
        gt_matched = [False] * len(gt_boxes)
        
        # Sort predictions by confidence (highest first)
        if pred_boxes:
            sorted_indices = np.argsort(pred_confs)[::-1]
            
            for pred_idx in sorted_indices:
                pred_box = pred_boxes[pred_idx]
                pred_cls = pred_classes[pred_idx]
                pred_conf = pred_confs[pred_idx]
                
                best_iou = 0
                best_gt_idx = -1
                
                # Find best matching ground truth box
                for gt_idx, (gt_box, gt_cls) in enumerate(zip(gt_xyxy, gt_classes)):
                    if gt_matched[gt_idx] or gt_cls != pred_cls:
                        continue
                    
                    iou = calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                # Determine if it's TP or FP
                if best_iou >= iou_threshold and best_gt_idx != -1:
                    # True Positive
                    class_metrics[pred_cls]['tp'] += 1
                    gt_matched[best_gt_idx] = True
                    is_tp = 1
                else:
                    # False Positive
                    class_metrics[pred_cls]['fp'] += 1
                    is_tp = 0
                
                # Store for precision-recall curve
                class_metrics[pred_cls]['confidences'].append(pred_conf)
                
                # Calculate running precision and recall
                current_tp = class_metrics[pred_cls]['tp']
                current_fp = class_metrics[pred_cls]['fp']
                total_gt_for_class = sum(1 for gt_cls in gt_classes if gt_cls == pred_cls) + \
                     class_metrics[pred_cls]['fn']  # Add previous FNs
                
                precision = current_tp / (current_tp + current_fp) if (current_tp + current_fp) > 0 else 0
                recall = current_tp / max(1, total_gt_for_class) if total_gt_for_class > 0 else 0
                
                class_metrics[pred_cls]['precisions'].append(precision)
                class_metrics[pred_cls]['recalls'].append(recall)
        
        # Count False Negatives (unmatched ground truth boxes)
        for gt_idx, (gt_cls, matched) in enumerate(zip(gt_classes, gt_matched)):
            if not matched:
                class_metrics[gt_cls]['fn'] += 1
    
    # Calculate final metrics for each class
    class_results = {}
    overall_metrics = {'tp': 0, 'fp': 0, 'fn': 0}
    
    for cls in range(num_classes):
        tp = class_metrics[cls]['tp']
        fp = class_metrics[cls]['fp']
        fn = class_metrics[cls]['fn']
        
        overall_metrics['tp'] += tp
        overall_metrics['fp'] += fp
        overall_metrics['fn'] += fn
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate AP
        ap = 0
        if class_metrics[cls]['precisions'] and class_metrics[cls]['recalls']:
            precisions = np.array(class_metrics[cls]['precisions'])
            recalls = np.array(class_metrics[cls]['recalls'])
            ap = calculate_ap(precisions, recalls)
        
        class_results[cls] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'ap': ap,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'support': tp + fn  # Number of ground truth instances
        }
    
    # Calculate overall metrics
    overall_tp = overall_metrics['tp']
    overall_fp = overall_metrics['fp']
    overall_fn = overall_metrics['fn']
    
    overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
    overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    # Calculate mAP
    ap_values = [class_results[cls]['ap'] for cls in range(num_classes) if class_results[cls]['support'] > 0]
    map_score = np.mean(ap_values) if ap_values else 0
    
    # Calculate accuracy (TP / (TP + FP + FN))
    total_instances = overall_tp + overall_fp + overall_fn
    accuracy = overall_tp / total_instances if total_instances > 0 else 0
    
    overall_results = {
        'precision': overall_precision,
        'recall': overall_recall,
        'f1_score': overall_f1,
        'map': map_score,
        'accuracy': accuracy,
        'total_images': len(predictions),
        'total_gt_boxes': total_gt_boxes,
        'total_pred_boxes': total_pred_boxes,
        'tp': overall_tp,
        'fp': overall_fp,
        'fn': overall_fn
    }
    
    return class_results, overall_results


def save_detection_results(results, save_path, img_name, save_txt=False, save_json_data=None):
    """Save detection results in various formats"""
    
    if save_txt:
        txt_path = os.path.join(save_path, 'labels', f"{Path(img_name).stem}.txt")
        os.makedirs(os.path.dirname(txt_path), exist_ok=True)
        
        with open(txt_path, 'w') as f:
            if results.boxes is not None:
                for box in results.boxes:
                    cls = int(box.cls.cpu().numpy()[0])
                    conf = float(box.conf.cpu().numpy()[0])
                    
                    # Convert to YOLO format
                    img_h, img_w = results.orig_shape
                    x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
                    
                    x_center = (x1 + x2) / 2 / img_w
                    y_center = (y1 + y2) / 2 / img_h
                    width = (x2 - x1) / img_w
                    height = (y2 - y1) / img_h
                    
                    f.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {conf:.6f}\n")


def annotate_and_save_image(image, results, save_path, img_name, line_thickness=3, hide_labels=False, hide_conf=False):
    """Annotate image with detection results and save"""
    annotator = Annotator(image, line_width=line_thickness, example=str(results.names))
    
    if results.boxes is not None:
        for box in results.boxes:
            cls = int(box.cls.cpu().numpy()[0])
            conf = float(box.conf.cpu().numpy()[0])
            x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
            
            # Create label
            if hide_labels and hide_conf:
                label = None
            elif hide_labels:
                label = f"{conf:.2f}"
            elif hide_conf:
                label = f"{results.names[cls]}"
            else:
                label = f"{results.names[cls]} {conf:.2f}"
            
            # Annotate
            annotator.box_label([x1, y1, x2, y2], label, color=colors(cls, True))
    
    # Save annotated image
    output_path = os.path.join(save_path, 'images', img_name)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, annotator.result())


def save_metrics_results(class_results, overall_results, class_names, save_path):
    """Save comprehensive metrics results"""
    
    # Save detailed text summary
    metrics_path = os.path.join(save_path, "metrics_summary.txt")
    with open(metrics_path, 'w') as f:
        f.write("YOLOv8-AM DETECTION METRICS SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("OVERALL METRICS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Precision:     {overall_results['precision']:.4f}\n")
        f.write(f"Recall:        {overall_results['recall']:.4f}\n")
        f.write(f"F1-Score:      {overall_results['f1_score']:.4f}\n")
        f.write(f"mAP:           {overall_results['map']:.4f}\n")
        f.write(f"Accuracy:      {overall_results['accuracy']:.4f}\n\n")
        
        f.write("CONFUSION MATRIX:\n")
        f.write("-" * 30 + "\n")
        f.write(f"True Positives:  {overall_results['tp']}\n")
        f.write(f"False Positives: {overall_results['fp']}\n")
        f.write(f"False Negatives: {overall_results['fn']}\n\n")
        
        f.write("DATASET SUMMARY:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total Images:           {overall_results['total_images']}\n")
        f.write(f"Total GT Boxes:         {overall_results['total_gt_boxes']}\n")
        f.write(f"Total Predicted Boxes:  {overall_results['total_pred_boxes']}\n\n")
        
        f.write("PER-CLASS METRICS:\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AP':<10} {'Support':<10}\n")
        f.write("-" * 60 + "\n")
        
        for cls, metrics in class_results.items():
            class_name = class_names.get(cls, f"Class_{cls}")
            f.write(f"{class_name:<15} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
                   f"{metrics['f1_score']:<10.4f} {metrics['ap']:<10.4f} {metrics['support']:<10}\n")
    
    # Save CSV for analysis
    csv_path = os.path.join(save_path, "metrics_detailed.csv")
    csv_data = []
    
    # Add overall row
    csv_data.append({
        'Class': 'Overall',
        'Class_Name': 'All Classes',
        'Precision': overall_results['precision'],
        'Recall': overall_results['recall'],
        'F1_Score': overall_results['f1_score'],
        'AP': overall_results['map'],
        'TP': overall_results['tp'],
        'FP': overall_results['fp'],
        'FN': overall_results['fn'],
        'Support': overall_results['tp'] + overall_results['fn']
    })
    
    # Add per-class rows
    for cls, metrics in class_results.items():
        class_name = class_names.get(cls, f"Class_{cls}")
        csv_data.append({
            'Class': cls,
            'Class_Name': class_name,
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1_Score': metrics['f1_score'],
            'AP': metrics['ap'],
            'TP': metrics['tp'],
            'FP': metrics['fp'],
            'FN': metrics['fn'],
            'Support': metrics['support']
        })
    
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_path, index=False)
    
    print(f"\nMetrics saved:")
    print(f"  Summary: {metrics_path}")
    print(f"  CSV:     {csv_path}")


def print_metrics_summary(class_results, overall_results, class_names):
    """Print metrics summary to console"""
    print("\n" + "=" * 70)
    print("DETECTION METRICS SUMMARY")
    print("=" * 70)
    
    print(f"\nOVERALL PERFORMANCE:")
    print(f"  Precision:  {overall_results['precision']:.4f}")
    print(f"  Recall:     {overall_results['recall']:.4f}")
    print(f"  F1-Score:   {overall_results['f1_score']:.4f}")
    print(f"  mAP:        {overall_results['map']:.4f}")
    print(f"  Accuracy:   {overall_results['accuracy']:.4f}")
    
    print(f"\nDETECTION COUNTS:")
    print(f"  True Positives:   {overall_results['tp']}")
    print(f"  False Positives:  {overall_results['fp']}")
    print(f"  False Negatives:  {overall_results['fn']}")
    
    print(f"\nDATASET INFO:")
    print(f"  Images Processed:     {overall_results['total_images']}")
    print(f"  Ground Truth Boxes:   {overall_results['total_gt_boxes']}")
    print(f"  Predicted Boxes:      {overall_results['total_pred_boxes']}")
    
    if len(class_results) > 1:
        print(f"\nPER-CLASS PERFORMANCE:")
        print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AP':<10}")
        print("-" * 65)
        for cls, metrics in class_results.items():
            if metrics['support'] > 0:  # Only show classes with ground truth
                class_name = class_names.get(cls, f"Class_{cls}")
                print(f"{class_name:<15} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
                     f"{metrics['f1_score']:<10.4f} {metrics['ap']:<10.4f}")
    
    print("=" * 70)


def save_json_results(all_results, save_path):
    """Save detection results in JSON format"""
    json_path = os.path.join(save_path, "detections.json")
    
    json_data = []
    for img_path, results in all_results.items():
        img_name = Path(img_path).name
        detections = []
        
        if results.boxes is not None:
            for box in results.boxes:
                cls = int(box.cls.cpu().numpy()[0])
                conf = float(box.conf.cpu().numpy()[0])
                x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0].tolist()
                
                detections.append({
                    "class_id": cls,
                    "class_name": results.names[cls],
                    "confidence": conf,
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                })
        
        json_data.append({
            "image": img_name,
            "image_path": str(img_path),
            "detections": detections,
            "detection_count": len(detections)
        })
    
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"JSON results saved to: {json_path}")


def run_inference(model, image_files, args):
    """Run inference on all images"""
    all_results = {}
    inference_times = []
    
    print(f"\nRunning inference on {len(image_files)} images...")
    
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            # Run inference
            start_time = time.time()
            
            results = model(
                img_path,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                classes=args.classes,
                agnostic_nms=args.agnostic_nms,
                augment=args.augment
            )[0]  # Get first (and only) result
            
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            inference_times.append(inference_time)
            
            # Store results
            all_results[img_path] = results
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            # Skip numpy/tensorboard compatibility errors
            if "np.object" in str(e) or "numpy" in str(e).lower():
                print(f"  Skipping numpy compatibility error")
            continue
    
    return all_results, inference_times


def main():
    """Main execution function"""
    # Parse arguments
    args = parse_arguments()
    
    # Print configuration
    print("YOLOv8-AM Fracture Detection with Comprehensive Metrics")
    print("=" * 60)
    print(f"Model:                    {args.model}")
    print(f"Source:                   {args.source}")
    print(f"Labels Directory:         {args.labels_dir or 'Not provided'}")
    print(f"Image Size:               {args.imgsz}")
    print(f"Confidence Threshold:     {args.conf}")
    print(f"IoU Threshold (NMS):      {args.iou}")
    print(f"IoU Threshold (Metrics):  {args.iou_thres_metrics}")
    print(f"Device:                   {args.device or 'auto-detect'}")
    print(f"Calculate Metrics:        {args.save_metrics}")
    print("-" * 60)
    
    # Setup device
    device = setup_device(args.device)
    
    # Load model
    model = load_model(args.model, device)
    if model is None:
        return
    
    # Get image files
    image_files = get_image_files(args.source)
    if not image_files:
        print("No valid image files found!")
        return
    
    print(f"Found {len(image_files)} image(s)")
    
    # Create output directory
    save_dir = create_output_directory(args.project, args.name, args.exist_ok)
    print(f"Results will be saved to: {save_dir}")
    
    # Run inference
    all_results, inference_times = run_inference(model, image_files, args)
    
    if not all_results:
        print("No results obtained. Check for errors above.")
        return
    
    print(f"Successfully processed {len(all_results)}/{len(image_files)} images")
    
    # Save detection outputs
    if args.save_txt or args.save_img:
        print("Saving detection results...")
        
        for img_path, results in tqdm(all_results.items(), desc="Saving results"):
            img_name = Path(img_path).name
            
            # Save text results
            if args.save_txt:
                save_detection_results(results, save_dir, img_name, save_txt=True)
            
            # Save annotated images
            if args.save_img:
                try:
                    image = cv2.imread(img_path)
                    if image is not None:
                        annotate_and_save_image(
                            image, results, save_dir, img_name,
                            args.line_thickness, args.hide_labels, args.hide_conf
                        )
                except Exception as e:
                    print(f"Error saving annotated image for {img_name}: {e}")
    
    # Save JSON results
    if args.save_json:
        save_json_results(all_results, save_dir)
    
    # Calculate metrics
    if args.save_metrics and args.labels_dir:
        print("\nCalculating comprehensive metrics...")
        
        # Load ground truth
        ground_truths = load_ground_truth_labels(args.labels_dir, list(all_results.keys()))
        
        if not any(gt for gt in ground_truths.values()):
            print("Warning: No ground truth labels found. Cannot calculate metrics.")
        else:
            # Calculate detailed metrics
            class_results, overall_results = calculate_comprehensive_metrics(
                all_results, ground_truths, len(model.names), args.iou_thres_metrics
            )
            
            # Print and save metrics
            print_metrics_summary(class_results, overall_results, model.names)
            save_metrics_results(class_results, overall_results, model.names, save_dir)
    else:
        # Print basic statistics
        print("\n" + "=" * 50)
        print("BASIC DETECTION STATISTICS")
        print("=" * 50)
        
        total_images = len(all_results)
        images_with_detections = 0
        total_detections = 0
        confidence_scores = []
        class_counts = {}
        
        for results in all_results.values():
            if results.boxes is not None and len(results.boxes) > 0:
                images_with_detections += 1
                total_detections += len(results.boxes)
                
                for box in results.boxes:
                    conf = float(box.conf.cpu().numpy()[0])
                    cls = int(box.cls.cpu().numpy()[0])
                    class_name = results.names[cls]
                    
                    confidence_scores.append(conf)
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print(f"Total images processed:     {total_images}")
        print(f"Images with detections:     {images_with_detections}")
        print(f"Detection rate:             {images_with_detections/total_images:.2%}")
        print(f"Total detections:           {total_detections}")
        print(f"Average detections/image:   {total_detections/total_images:.2f}")
        
        if confidence_scores:
            print(f"Average confidence:         {np.mean(confidence_scores):.3f}")
            print(f"Confidence range:           [{np.min(confidence_scores):.3f}, {np.max(confidence_scores):.3f}]")
            
            if class_counts:
                print(f"\nClass Distribution:")
                for class_name, count in sorted(class_counts.items()):
                    percentage = count / total_detections * 100
                    print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        print("=" * 50)
        
        if not args.labels_dir:
            print(f"\nNote: For detailed metrics (precision, recall, F1, mAP), provide ground truth labels with --labels-dir")
    
    # Print timing information
    if inference_times:
        avg_time = np.mean(inference_times)
        total_time = sum(inference_times) / 1000
        print(f"\nTiming Information:")
        print(f"  Average inference time: {avg_time:.1f} ms")
        print(f"  Total inference time:   {total_time:.2f} s")
        print(f"  Images per second:      {len(inference_times)/total_time:.1f}")
    
    # Save summary
    summary_path = save_dir / 'summary.txt'
    with open(summary_path, 'w') as f:
        f.write("YOLOv8-AM Detection Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Source: {args.source}\n")
        f.write(f"Labels: {args.labels_dir or 'Not provided'}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Configuration: imgsz={args.imgsz}, conf={args.conf}, iou={args.iou}\n")
        f.write(f"Images processed: {len(all_results)}/{len(image_files)}\n")
        
        if args.save_metrics and args.labels_dir and 'overall_results' in locals():
            f.write(f"\nMetrics (IoU={args.iou_thres_metrics}):\n")
            f.write(f"Precision: {overall_results['precision']:.4f}\n")
            f.write(f"Recall: {overall_results['recall']:.4f}\n")
            f.write(f"F1-Score: {overall_results['f1_score']:.4f}\n")
            f.write(f"mAP: {overall_results['map']:.4f}\n")
            f.write(f"Accuracy: {overall_results['accuracy']:.4f}\n")
        
        if inference_times:
            f.write(f"\nTiming:\n")
            f.write(f"Average inference time: {np.mean(inference_times):.1f} ms\n")
            f.write(f"Total processing time: {sum(inference_times)/1000:.2f} s\n")
    
    print(f"\nSummary saved to: {summary_path}")
    print("Detection completed successfully!")


if __name__ == '__main__':
    main()