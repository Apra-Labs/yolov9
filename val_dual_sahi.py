import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import json

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.augmentations import (letterbox)
from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.general import (LOGGER, TQDM_BAR_FORMAT, Profile, check_dataset, check_img_size, check_requirements,
                           check_yaml, coco80_to_coco91_class, colorstr, increment_path, non_max_suppression,
                           print_args, scale_boxes, xywh2xyxy, xyxy2xywh)
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, smart_inference_mode

def resize_with_letterbox_bbox(bbox, original_size, target_size):
    """
    Args:
        bbox: (x1, y1, x2, y2) tuple in original image.
        original_size: (height, width) of the original image.
        target_size: (height, width) of the target image after letterboxing.
    
    Returns:
        New bbox (x1, y1, x2, y2) after resizing and letterboxing.
    """

    orig_h, orig_w = original_size
    target_h, target_w = target_size

    # Calculate scale factors
    scale = min(target_w / orig_w, target_h / orig_h)

    # Compute the size after scaling
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    # Calculate padding (letterbox) on each side
    pad_w = (target_w - new_w) // 2
    pad_h = (target_h - new_h) // 2

    x1, y1, x2, y2 = bbox

    # Scale the coordinates
    x1 = x1 * scale + pad_w
    y1 = y1 * scale + pad_h
    x2 = x2 * scale + pad_w
    y2 = y2 * scale + pad_h

    return (int(x1), int(y1), int(x2), int(y2))

def Letterbox(image, target_size=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
    """
    Resize and pad image to fit target size while keeping aspect ratio.
    """
    shape = image.shape[:2]  # current shape [height, width]
    target_w, target_h = target_size

    # Scale ratio (new / old)
    scale = min(target_w / shape[1], target_h / shape[0])
    if not scaleup:
        scale = min(scale, 1.0)

    # Compute new unpadded size
    new_w = int(round(shape[1] * scale))
    new_h = int(round(shape[0] * scale))

    # Compute padding
    dw = target_w - new_w
    dh = target_h - new_h
    if auto:  # minimum rectangle, multiple of 32
        dw = np.mod(dw, 32)
        dh = np.mod(dh, 32)
    elif scaleFill:  # stretch to fill the shape
        new_w, new_h = target_w, target_h
        dw, dh = 0, 0
        scale = (target_w / shape[1], target_h / shape[0])

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Add border
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded_image = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return padded_image, (scale, scale), (dw, dh)


def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')


def save_one_json(predn, jdict, path, class_map):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({
            'image_id': image_id,
            'category_id': class_map[int(p[5])],
            'bbox': [round(x, 3) for x in b],
            'score': round(p[4], 5)})


def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


@smart_inference_mode()
def run(
        data,
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        class_conf_thres=None,  # class-specific confidence thresholds
        iou_thres=0.7,  # NMS IoU threshold
        max_det=300,  # maximum detections per image
        task='val',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        workers=8,  # max dataloader workers (per RANK in DDP mode)
        single_cls=False,  # treat as single-class dataset
        augment=False,  # augmented inference
        verbose=False,  # verbose output
        save_txt=False,  # save results to *.txt
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_json=False,  # save a COCO-JSON results file
        project=ROOT / 'runs/val',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        min_items=0,  # Experimental
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        callbacks=Callbacks(),
        compute_loss=None,
        use_sahi=False,  # whether to use SAHI
        sahi_slice_height=512,  # SAHI slice height
        sahi_slice_width=512,  # SAHI slice width
        sahi_overlap_height_ratio=0.2,  # SAHI overlap height ratio
        sahi_overlap_width_ratio=0.2,  # SAHI overlap width ratio
        sahi_blob_area_threshold=0,  # minimum blob area to consider for SAHI
        use_custom_rois=False,
        top_k=1,
        blob_defects_info=None,
):
    image_ids = []
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model.half() if half else model.float()
    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        half = model.fp16  # FP16 supported on limited backends with CUDA
        if engine:
            batch_size = model.batch_size
        else:
            device = model.device
            if not (pt or jit):
                batch_size = 1  # export.py models default to batch-size 1
                LOGGER.info(f'Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models')

        names = model.names if hasattr(model, 'names') else model.module.names  # get class names
        if isinstance(names, (list, tuple)):  # old format
            names = dict(enumerate(names))

        # Initialize SAHI model if needed
        category_mapping = {str(k): v for k, v in names.items()}
        sahi_model = None
        if use_sahi:
            sahi_model = AutoDetectionModel.from_pretrained(
                model_type='yolov9pytorch',
                model_path=weights[0],
                confidence_threshold=conf_thres,
                device=device,
                category_mapping=category_mapping,
            )

        if blob_defects_info:
            with open(blob_defects_info, "r") as f:
                blob_data = json.load(f)

        # Data
        data = check_dataset(data)  # check

    # Configure
    model.eval()
    cuda = device.type != 'cpu'
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith(f'val2017.txt')  # COCO dataset
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if not training:
        if pt and not single_cls:  # check --weights are trained on --data
            ncm = model.model.nc
            assert ncm == nc, f'{weights} ({ncm} classes) trained on different --data than what you passed ({nc} ' \
                              f'classes). Pass correct combination of --weights and --data that are trained together.'
        model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # warmup
        pad, rect = (0.0, False) if task == 'speed' else (0.5, pt)  # square inference for benchmarks
        task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        dataloader = create_dataloader(data[task],
                                       imgsz,
                                       batch_size,
                                       stride,
                                       single_cls,
                                       pad=pad,
                                       rect=rect,
                                       workers=workers,
                                       min_items=min_items,
                                       prefix=colorstr(f'{task}: '))[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = model.names if hasattr(model, 'names') else model.module.names  # get class names
    if isinstance(names, (list, tuple)):  # old format
        names = dict(enumerate(names))
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95')
    tp, fp, p, r, f1, mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    dt = Profile(), Profile(), Profile()  # profiling times
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    callbacks.run('on_val_start')
    pbar = tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)  # progress bar
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        callbacks.run('on_val_batch_start')
        with dt[0]:
            if cuda:
                im = im.to(device, non_blocking=True)
                targets = targets.to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            nb, _, height, width = im.shape  # batch size, channels, height, width

        # Inference
        with dt[1]:
            preds, train_out = model(im) if compute_loss else (model(im, augment=augment), None)

        # Loss
        if compute_loss:
            preds = preds[1]
        else:
            preds = preds[0][1]

        # NMS
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        with dt[2]:
            preds = non_max_suppression(preds,
                                        conf_thres,
                                        iou_thres,
                                        labels=lb,
                                        multi_label=True,
                                        agnostic=single_cls,
                                        max_det=max_det)

        # Metrics
        for si, pred in enumerate(preds):
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            path, shape = Path(paths[si]), shapes[si][0]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            seen += 1

            # Blob Data
            if not path.name.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                continue

            idx = path.name
            # Find the corresponding blob in the JSON data
            blob_info = next((b for b in blob_data.get("Blobs", []) if b.get("Index Count String") == idx), None)
            if not blob_info:
                print(f"No blob found for Index Count String {idx}")
            
            # Apply SAHI if enabled
            if use_sahi and (len(pred) == 0 or blob_info):
                # Get initial detections
                initial_detections = []
                for *xyxy, conf, cls in pred:
                    x1, y1, x2, y2 = xyxy  # xyxy is already float tensor
                    area = (x2 - x1) * (y2 - y1)
                    if area >= sahi_blob_area_threshold:
                        initial_detections.append((x1, y1, x2, y2, float(conf), int(cls)))

                # Letterbox the image for SAHI
                im0 = im.squeeze(0).permute(1, 2, 0).cpu().numpy()
                if im0.max() <= 1.0:
                    im0 = (im0 * 255).astype(np.uint8)
                else:
                    im0 = im0.astype(np.uint8)

                letterboxed_img, ratio, (dw, dh) = Letterbox(im0, target_size=(960, 960))
                letterbox_shape = im0.shape[:2]
                
                custom_rois = []
                if use_custom_rois:
                    for det in initial_detections:
                        x1, y1, x2, y2, conf, cls = det
                        roi = [x1, y1, x2, y2]
                        custom_rois.append(resize_with_letterbox_bbox(roi, (letterbox_shape[0], letterbox_shape[1]), (960, 960)))
                
                if blob_info:
                    for contour in blob_info.get("Contours", []):
                        top = int(float(contour["Top"]))
                        bottom = int(float(contour["Bottom"]))
                        left = int(float(contour["Left"]))
                        right = int(float(contour["Right"]))
                        x_min = min(left, right)
                        x_max = max(left, right)
                        y_min = min(top, bottom)
                        y_max = max(top, bottom)
                        roi = [x_min, y_min, x_max, y_max]
                        if (x_max - x_min) * (y_max - y_min) <= 1.0:
                            continue
                        custom_rois.append(resize_with_letterbox_bbox(roi, (letterbox_shape[0], letterbox_shape[1]), (960, 960)))

                # Get SAHI predictions
                sahi_result = get_sliced_prediction(
                    letterboxed_img,
                    sahi_model,
                    slice_height=sahi_slice_height,
                    slice_width=sahi_slice_width,
                    overlap_height_ratio=sahi_overlap_height_ratio,
                    overlap_width_ratio=sahi_overlap_width_ratio,
                    custom_rois=custom_rois
                )

                # Combine initial and SAHI detections
                combined_detections = initial_detections
                for detection in sahi_result.object_prediction_list:
                    # Convert SAHI bbox to original image coordinates
                    box = detection.bbox.to_xyxy()
                    
                    # Convert from letterboxed coordinates to original image coordinates
                    x1 = (box[0] - dw) / ratio[0]
                    y1 = (box[1] - dh) / ratio[1]
                    x2 = (box[2] - dw) / ratio[0]
                    y2 = (box[3] - dh) / ratio[1]
                    
                    # Ensure coordinates are within image bounds
                    x1 = max(0, min(x1, im0.shape[1]))
                    y1 = max(0, min(y1, im0.shape[0]))
                    x2 = max(0, min(x2, im0.shape[1]))
                    y2 = max(0, min(y2, im0.shape[0]))
                    
                    conf = detection.score.value
                    cls = detection.category.id
                    combined_detections.append((x1, y1, x2, y2, conf, cls))

                # Convert to tensor format and ensure proper shape
                if combined_detections:
                    pred = torch.tensor(combined_detections, device=device)
                    if len(pred.shape) == 1:
                        pred = pred.unsqueeze(0)

            npr = pred.shape[0]
            
            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                    if plots:
                        confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                continue

            # Apply class-specific confidence thresholds
            if class_conf_thres is not None:
                mask = torch.ones(len(pred), dtype=torch.bool)
                for i, (*xyxy, conf, cls) in enumerate(pred):
                    cls_thres = class_conf_thres.get(int(cls), conf_thres)
                    if conf < cls_thres:
                        mask[i] = False
                pred = pred[mask]

            # Sort by confidence score in descending order
            sorted_indices = torch.argsort(pred[:, 4], descending=True)
            # Take top_k
            top_n = min(top_k, len(pred))
            top_detections = pred[sorted_indices[:top_n]]

            
            # Combine all top detections
            if top_detections:
                pred = torch.cat(top_detections, dim=0)
            
            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)
            
            stats_temp = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
            corrects, confs, pcls, tcls = stats_temp
            n = len(corrects)
            image_ids.append(si)

            # Check for mismatch
            if not (corrects.shape[0] == confs.shape[0] == pcls.shape[0]):
                print("❌ Stats shape mismatch detected:")
                print(f"  corrects: {corrects.shape}")
                print(f"  confs:    {confs.shape}")
                print(f"  pcls:     {pcls.shape}")

                # Trace back which image caused mismatch
                for i, (c, conf, p, t, path) in enumerate(zip(stats[0], stats[1], stats[2], stats[3], image_ids)):
                    if c.shape[0] != 1 and (c.shape[0] != conf.shape[0] or conf.shape[0] != p.shape[0]):
                        print(f"❗ Shape mismatch at index {i} -> Image: {path}")
                        print(f"   correct.shape: {c.shape}, conf: {conf.shape}, pcls: {p.shape}")
                        break

                raise ValueError("Stats dimensions mismatch.")

            # Save/log
            if save_txt:
                save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / f'{path.stem}.txt')
            if save_json:
                save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary
            callbacks.run('on_val_image_end', pred, predn, path, names, im[si])

        # Plot images
        if plots and batch_i < 3:
            plot_images(im, targets, paths, save_dir / f'val_batch{batch_i}_labels.jpg', names)  # labels
            plot_images(im, output_to_target(preds), paths, save_dir / f'val_batch{batch_i}_pred.jpg', names)  # pred

        callbacks.run('on_val_batch_end', batch_i, im, targets, paths, shapes, preds)

    # Compute metrics after all batches
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class

    # Print results
    pf = '%22s' + '%11i' * 2 + '%11.3g' * 4  # print format
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
    if nt.sum() == 0:
        LOGGER.warning(f'WARNING ⚠️ no labels found in {task} set, can not compute metrics without labels')

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        callbacks.run('on_val_end', nt, tp, fp, p, r, f1, ap, ap50, ap_class, confusion_matrix)

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = str(Path(data.get('path', '../coco')) / 'annotations/instances_val2017.json')  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        LOGGER.info(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            check_requirements('pycocotools')
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.im_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            LOGGER.info(f'pycocotools unable to run: {e}')

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolo.pt', help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--class-conf-thres', type=str, default=None, help='class-specific confidence thresholds in format "class_id:threshold,class_id:threshold"')
    parser.add_argument('--iou-thres', type=float, default=0.7, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=300, help='maximum detections per image')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--min-items', type=int, default=0, help='Experimental')
    parser.add_argument('--use-sahi', action='store_true', help='use SAHI for improved detection')
    parser.add_argument('--sahi-slice-height', type=int, default=400, help='SAHI slice height')
    parser.add_argument('--sahi-slice-width', type=int, default=400, help='SAHI slice width')
    parser.add_argument('--sahi-overlap-height-ratio', type=float, default=0.2, help='SAHI overlap height ratio')
    parser.add_argument('--sahi-overlap-width-ratio', type=float, default=0.2, help='SAHI overlap width ratio')
    parser.add_argument('--sahi-blob-area-threshold', type=int, default=0, help='minimum blob area to consider for SAHI')
    parser.add_argument('--use-custom-rois', action='store_true', help='whether to use custom rois when performing SAHI')
    parser.add_argument('--top-k', type=int, default=1, help='take top k predictions')
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    print_args(vars(opt))

    # Parse class-specific confidence thresholds
    if opt.class_conf_thres:
        class_conf_thres = {}
        for item in opt.class_conf_thres.split(','):
            class_id, threshold = item.split(':')
            class_conf_thres[int(class_id)] = float(threshold)
        opt.class_conf_thres = class_conf_thres
    else:
        opt.class_conf_thres = None

    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt) 