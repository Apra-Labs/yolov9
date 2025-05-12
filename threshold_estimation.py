import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import itertools
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import gc
import psutil
import math
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import create_dataloader
from utils.general import (LOGGER, TQDM_BAR_FORMAT, check_dataset, check_img_size, check_requirements,
                         check_yaml, colorstr, increment_path, non_max_suppression, print_args)
from utils.metrics import ap_per_class
from utils.torch_utils import select_device, smart_inference_mode
import val_dual_sahi as validate

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov9-c.pt', help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=300, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=ROOT / 'runs/grid-search', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_false', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--min-thres', type=float, default=0.1, help='minimum confidence threshold')
    parser.add_argument('--max-thres', type=float, default=0.9, help='maximum confidence threshold')
    parser.add_argument('--metric', type=str, default='f1', choices=['f1', 'map50', 'map'], help='metric to optimize')
    parser.add_argument('--n-jobs', type=int, default=4, help='number of parallel jobs')
    parser.add_argument('--max-memory-percent', type=float, default=90.0, help='maximum memory usage percentage before cleanup')
    parser.add_argument('--max-iterations', type=int, default=1, help='maximum number of optimization iterations')
    parser.add_argument('--class-weights', type=str, default=None, help='JSON file containing class weights for optimization')
    parser.add_argument('--blob-defects', type=str, default=None, help='JSON file containing blob defect information')
    return parser.parse_args()

def get_memory_usage():
    """Get current memory usage percentage"""
    return psutil.virtual_memory().percent

class OptimizationCallback:
    def __init__(self):
        self.history = []
        self.metrics = []
    
    def __call__(self, xk):
        self.history.append(xk.copy())
        # Calculate and store metric value
        metric_value = -objective_function(xk, args)  # Convert back to positive
        self.metrics.append(metric_value)
        # Print current thresholds and metric
        LOGGER.info(f'\nCurrent thresholds (iteration {len(self.history)}):')
        for i, thres in enumerate(xk):
            LOGGER.info(f'Class {i}: {thres:.3f}')
        LOGGER.info(f'Current {args.metric.upper()}: {metric_value:.4f}')
        return False

def evaluate_thresholds(args, thresholds):
    """Evaluate model with given class confidence thresholds"""
    try:
        # Convert thresholds to dictionary
        class_conf_thres = {i: float(t) for i, t in enumerate(thresholds)}
        
        # Print thresholds being evaluated
        LOGGER.info('\nEvaluating thresholds:')
        for cls, thres in class_conf_thres.items():
            LOGGER.info(f'Class {cls}: {thres:.3f}')
        
        # Load blob defect information if provided
        blob_defects = None
        if args.blob_defects:
            try:
                with open(args.blob_defects, 'r') as f:
                    blob_defects = json.load(f)
                LOGGER.info(f'Loaded blob defects from: {args.blob_defects}')
            except Exception as e:
                LOGGER.error(f'Error loading blob defects: {str(e)}')
                blob_defects = None
        
        results, maps, _ = validate.run(
            data=args.data,
            weights=args.weights,
            batch_size=args.batch_size,
            imgsz=args.imgsz,
            conf_thres=args.conf_thres,
            class_conf_thres=class_conf_thres,
            iou_thres=args.iou_thres,
            max_det=args.max_det,
            device=args.device,
            workers=args.workers,
            project=args.project,
            name=args.name,
            exist_ok=args.exist_ok,
            half=args.half,
            dnn=args.dnn,
            verbose=False,
            save_txt=False,
            save_json=False,
            save_hybrid=False,
            save_conf=False,
            plots=True,
            blob_defects=blob_defects  # Pass blob defects to validation
        )
        
        # Extract metrics
        mp, mr, map50, map = results[:4]
        f1 = 2 * (mp * mr) / (mp + mr) if (mp + mr) > 0 else 0
        
        # Load class weights if provided
        class_weights = None
        if args.class_weights:
            try:
                with open(args.class_weights, 'r') as f:
                    class_weights = json.load(f)
                LOGGER.info(f'Loaded class weights: {class_weights}')
            except Exception as e:
                LOGGER.error(f'Error loading class weights: {str(e)}')
                class_weights = None
        
        # Calculate weighted metric if class weights are provided
        if class_weights:
            weighted_metric = 0
            total_weight = 0
            
            # maps contains per-class AP values
            for cls_idx, weight in class_weights.items():
                cls_idx = int(cls_idx)
                if cls_idx < len(maps):
                    if args.metric == 'f1':
                        # For F1, we need to calculate per-class F1
                        # We can use precision and recall from the confusion matrix
                        tp = results[4][cls_idx]  # True positives for class
                        fp = results[5][cls_idx]  # False positives for class
                        fn = results[6][cls_idx]  # False negatives for class
                        
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                        cls_metric = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    elif args.metric == 'map50':
                        cls_metric = maps[cls_idx]  # mAP50 for class
                    else:  # map
                        cls_metric = maps[cls_idx]  # mAP for class
                    
                    weighted_metric += weight * cls_metric
                    total_weight += weight
            
            if total_weight > 0:
                weighted_metric /= total_weight
                LOGGER.info(f'Weighted {args.metric.upper()}: {weighted_metric:.4f}')
                return -weighted_metric  # Negative because we're minimizing
        
        # If no weights or error in loading weights, use overall metric
        if args.metric == 'f1':
            return -f1  # Negative because we're minimizing
        elif args.metric == 'map50':
            return -map50
        else:  # map
            return -map
    except Exception as e:
        LOGGER.error(f"Error evaluating thresholds {thresholds}: {str(e)}")
        return float('inf')  # Return infinity for failed evaluations

def objective_function(thresholds, args):
    """Objective function for optimization"""
    return evaluate_thresholds(args, thresholds)

def plot_results(save_dir, history, metrics, best_thresholds, best_metric, metric_name, class_names):
    """Plot optimization results"""
    # Set style
    plt.style.use('classic')
    
    # 1. Plot optimization convergence
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(history)), metrics, 'b-', label='Metric Value')
    plt.xlabel('Iteration')
    plt.ylabel(metric_name.upper())
    plt.title(f'Optimization Convergence ({metric_name.upper()})')
    plt.grid(True)
    plt.legend()
    plt.savefig(save_dir / 'optimization_convergence.png')
    plt.close()
    
    # 2. Plot best thresholds per class
    plt.figure(figsize=(12, 6))
    classes = list(range(len(best_thresholds)))
    plt.bar(classes, best_thresholds, color='skyblue')
    plt.xlabel('Class')
    plt.ylabel('Confidence Threshold')
    plt.title('Best Confidence Thresholds per Class')
    plt.xticks(classes, [class_names[i] if i in class_names else f'Class {i}' for i in classes], rotation=45)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(save_dir / 'best_thresholds.png')
    plt.close()
    
    # 3. Plot threshold distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(best_thresholds, bins=20, kde=True)
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Count')
    plt.title('Distribution of Best Confidence Thresholds')
    plt.grid(True)
    plt.savefig(save_dir / 'threshold_distribution.png')
    plt.close()
    
    # 4. Plot threshold vs class index scatter
    plt.figure(figsize=(12, 6))
    plt.scatter(classes, best_thresholds, c=best_thresholds, cmap='viridis', s=100)
    plt.colorbar(label='Threshold Value')
    plt.xlabel('Class Index')
    plt.ylabel('Confidence Threshold')
    plt.title('Confidence Thresholds vs Class Index')
    plt.grid(True)
    plt.xticks(classes, [class_names[i] if i in class_names else f'Class {i}' for i in classes], rotation=45)
    plt.tight_layout()
    plt.savefig(save_dir / 'threshold_scatter.png')
    plt.close()
    
    # 5. Plot threshold evolution for each class
    plt.figure(figsize=(15, 8))
    history_array = np.array(history)
    for i in range(history_array.shape[1]):
        plt.plot(range(len(history)), history_array[:, i], label=f'Class {i}')
    plt.xlabel('Iteration')
    plt.ylabel('Threshold Value')
    plt.title('Threshold Evolution During Optimization')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_dir / 'threshold_evolution.png')
    plt.close()

def grid_search(args):
    """Perform efficient search for optimal class confidence thresholds"""
    # Load model to get number of classes
    device = select_device(args.device)
    model = DetectMultiBackend(args.weights, device=device, dnn=args.dnn, data=args.data, fp16=args.half)
    
    # Get number of classes from data config
    data_dict = check_dataset(args.data)
    nc = int(data_dict['nc'])  # number of classes
    
    # Get class names
    class_names = data_dict.get('names', {})
    
    # Create results directory
    save_dir = increment_path(Path(args.project) / args.name, exist_ok=args.exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate multiple random initial points
    n_initial_points = 5
    initial_points = []
    for _ in range(n_initial_points):
        # Random initialization between min and max threshold
        x0 = np.random.uniform(args.min_thres, args.max_thres, nc)
        initial_points.append(x0)
    
    # Bounds for each threshold
    bounds = [(args.min_thres, args.max_thres) for _ in range(nc)]
    
    # Setup callback to track optimization history
    callback = OptimizationCallback()
    
    # Optimization options
    options = {
        'maxiter': args.max_iterations,
        'disp': True,
        'ftol': 1e-4,  # Function tolerance
        'gtol': 1e-4,  # Gradient tolerance
    }
    
    # Try different optimization methods
    methods = ['Nelder-Mead', 'Powell', 'L-BFGS-B']
    best_result = None
    best_metric = float('inf')
    
    LOGGER.info(f'Starting optimization for {nc} classes...')
    
    for method in methods:
        LOGGER.info(f'\nTrying optimization method: {method}')
        
        for i, x0 in enumerate(initial_points):
            LOGGER.info(f'\nInitial point {i+1}/{n_initial_points}:')
            for j, thres in enumerate(x0):
                LOGGER.info(f'Class {j}: {thres:.3f}')
            
            try:
                result = minimize(
                    objective_function,
                    x0,
                    args=(args,),
                    method=method,
                    bounds=bounds if method == 'L-BFGS-B' else None,
                    options=options,
                    callback=callback
                )
                
                if result.fun < best_metric:
                    best_metric = result.fun
                    best_result = result
                    LOGGER.info(f'New best result found with {method}:')
                    LOGGER.info(f'Final metric: {-best_metric:.4f}')
                    for j, thres in enumerate(result.x):
                        LOGGER.info(f'Class {j}: {thres:.3f}')
                
            except Exception as e:
                LOGGER.error(f'Error with {method} optimization: {str(e)}')
                continue
    
    if best_result is None:
        raise RuntimeError('All optimization attempts failed')
    
    # Get best thresholds
    best_thresholds = best_result.x
    best_metric = -best_metric  # Convert back to positive (we minimized negative metric)
    
    # Save results
    best_thresholds_dict = {i: float(t) for i, t in enumerate(best_thresholds)}
    with open(save_dir / 'best_thresholds.json', 'w') as f:
        json.dump(best_thresholds_dict, f, indent=4)
    
    # Save optimization history
    history = {
        'iterations': best_result.nit,
        'function_calls': best_result.nfev,
        'success': best_result.success,
        'message': best_result.message,
        'best_thresholds': best_thresholds_dict,
        'best_metric': float(best_metric),
        'threshold_history': [x.tolist() for x in callback.history],
        'metric_history': callback.metrics,
        'optimization_method': best_result.message
    }
    with open(save_dir / 'optimization_history.json', 'w') as f:
        json.dump(history, f, indent=4)
    
    # Plot results
    plot_results(save_dir, callback.history, callback.metrics, best_thresholds, best_metric, args.metric, class_names)
    
    # Print results
    LOGGER.info(f'\nBest {args.metric.upper()} score: {best_metric:.4f}')
    LOGGER.info('Best class confidence thresholds:')
    for cls, thres in best_thresholds_dict.items():
        cls_name = class_names.get(cls, f'Class {cls}')
        LOGGER.info(f'{cls_name}: {thres:.3f}')
    
    return best_thresholds_dict, best_metric

def main(opt):
    check_requirements(('torch', 'torchvision', 'psutil', 'scipy', 'matplotlib', 'seaborn'))
    best_thresholds, best_metric = grid_search(opt)
    return best_thresholds, best_metric

if __name__ == "__main__":
    opt = parse_opt()
    main(opt) 