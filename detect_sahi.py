import argparse
import os
import platform
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.slicing import slice_image

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
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

def letterbox(image, target_size=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
    """
    Resize and pad image to fit target size while keeping aspect ratio.

    Args:
        image (np.ndarray): Input image (BGR or RGB).
        target_size (tuple): Desired size in the format (width, height).
        color (tuple): Padding color. Default is gray (114,114,114).
        auto (bool): Use minimum rectangle that is a multiple of 32.
        scaleFill (bool): Stretch image to fill the target size (no aspect ratio preservation).
        scaleup (bool): Allow scaling up the image.

    Returns:
        padded_image: The resized and padded image.
        ratio: (width_scale, height_scale)
        (dw, dh): padding added on width and height
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

@smart_inference_mode()
def run(
        weights=ROOT / 'yolo.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # default confidence threshold
        class_conf_thres: Dict[int, float] = None,  # class-specific confidence thresholds
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        use_sahi=False,  # whether to use SAHI
        sahi_slice_height=512,  # SAHI slice height
        sahi_slice_width=512,  # SAHI slice width
        sahi_overlap_height_ratio=0.2,  # SAHI overlap height ratio
        sahi_overlap_width_ratio=0.2,  # SAHI overlap width ratio
        sahi_blob_area_threshold=0,  # minimum blob area to consider for SAHI
        use_custom_rois=False,
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

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

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)
            pred = pred[0][1]

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

            # Apply SAHI if enabled
            if use_sahi and len(det) > 0:
                # Get initial detections
                initial_detections = []
                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = map(int, xyxy)
                    area = (x2 - x1) * (y2 - y1)
                    if area >= sahi_blob_area_threshold:
                        initial_detections.append((x1, y1, x2, y2, float(conf), int(cls)))

                # Letterbox the image for SAHI
                letterboxed_img, ratio, (dw, dh) = letterbox(im0, target_size=(960, 960))
                shape = im0.shape[:2]

                custom_rois = []
                if use_custom_rois:
                    for det in initial_detections:
                        x1, y1, x2, y2, conf, cls = det
                        roi = [x1, y1, x2, y2]
                        custom_rois.append(resize_with_letterbox_bbox(roi, (shape[0], shape[1]), (960, 960)))

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
                    # Remove padding and scale back to original size
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
                    det = torch.tensor(combined_detections, device=model.device)
                    # Ensure det has the correct shape [N, 6] where N is number of detections
                    if len(det.shape) == 1:
                        det = det.unsqueeze(0)
                else:
                    det = torch.empty((0, 6), device=model.device)

            if len(det):
                # Apply class-specific confidence thresholds
                if class_conf_thres is not None:
                    mask = torch.ones(len(det), dtype=torch.bool)
                    for i, (*xyxy, conf, cls) in enumerate(det):
                        cls_thres = class_conf_thres.get(int(cls), conf_thres)
                        if conf < cls_thres:
                            mask[i] = False
                    det = det[mask]

                # Keep only top 3 detections per class
                top_detections = []
                for cls in det[:, 5].unique():
                    cls_mask = det[:, 5] == cls
                    cls_detections = det[cls_mask]
                    # Sort by confidence score in descending order
                    sorted_indices = torch.argsort(cls_detections[:, 4], descending=True)
                    # Take top 3 or all if less than 3
                    top_n = min(1, len(cls_detections))
                    top_detections.append(cls_detections[sorted_indices[:top_n]])
                
                # Combine all top detections
                if top_detections:
                    det = torch.cat(top_detections, dim=0)
                else:
                    det = torch.empty((0, 6), device=model.device)

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolo.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='default confidence threshold')
    parser.add_argument('--class-conf-thres', type=str, default=None, help='class-specific confidence thresholds in format "class_id:threshold,class_id:threshold"')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--use-sahi', action='store_true', help='use SAHI for improved detection')
    parser.add_argument('--sahi-slice-height', type=int, default=512, help='SAHI slice height')
    parser.add_argument('--sahi-slice-width', type=int, default=512, help='SAHI slice width')
    parser.add_argument('--sahi-overlap-height-ratio', type=float, default=0.2, help='SAHI overlap height ratio')
    parser.add_argument('--sahi-overlap-width-ratio', type=float, default=0.2, help='SAHI overlap width ratio')
    parser.add_argument('--sahi-blob-area-threshold', type=int, default=0, help='minimum blob area to consider for SAHI')
    parser.add_argument('--use-custom-rois', action='store_true', help='whether to use custom rois when performing SAHI')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    # Parse class-specific confidence thresholds
    if opt.class_conf_thres:
        class_conf_thres = {}
        for item in opt.class_conf_thres.split(','):
            class_id, threshold = item.split(':')
            class_conf_thres[int(class_id)] = float(threshold)
        opt.class_conf_thres = class_conf_thres
    else:
        opt.class_conf_thres = None

    print_args(vars(opt))
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt) 