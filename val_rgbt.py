# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Validate a trained YOLOv5 detection model on a detection dataset with RGB-T support.

Usage:
    $ python val_rgbt.py --weights yolov5s.pt --data coco128.yaml --img 640
    $ python val_rgbt.py --weights best.pt --data kaist-rgbt.yaml --rgbt --task test --save-json

RGB-T Usage:
    $ python val_rgbt.py --weights runs/train/yolov5s_nc4/weights/best.pt --data data/kaist-rgbt-fold1.yaml --rgbt --task test --save-json --single-cls --device 1
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.general import (
    LOGGER,
    TQDM_BAR_FORMAT,
    Profile,
    check_dataset,
    check_img_size,
    check_yaml,
    colorstr,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    xywh2xyxy,
    xyxy2xywh,
)
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, smart_inference_mode


def save_one_txt(predn, save_conf, shape, file):
    """Saves one detection result to a txt file in normalized xywh format, optionally including confidence."""
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, "a") as f:
            f.write(("%g " * len(line)).rstrip() % line + "\n")


def save_one_json(predn, jdict, path, index, class_map):
    """
    Saves one JSON detection result with image ID, category ID, bounding box, and score.

    Example: {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    """
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        if p[4] < 0.1:
            continue
        jdict.append(
            {
                "image_name": image_id,
                "image_id": int(index),
                "category_id": class_map[int(p[5])],
                "bbox": [round(x, 3) for x in b],
                "score": round(p[4], 5),
            }
        )


def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix.

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
    iou_thres=0.6,  # NMS IoU threshold
    max_det=300,  # maximum detections per image
    task="val",  # train, val, test, speed or study
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    workers=8,  # max dataloader workers (per RANK in DDP mode)
    single_cls=False,  # treat as single-class dataset
    augment=False,  # augmented inference
    verbose=False,  # verbose output
    save_txt=False,  # save results to *.txt
    save_hybrid=False,  # save label+prediction hybrid results to *.txt
    save_conf=False,  # save confidences in --save-txt labels
    save_json=False,  # save a COCO-JSON results file
    project=ROOT / "runs/val",  # save to project/name
    name="exp",  # save to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    half=True,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    model=None,
    dataloader=None,
    save_dir=Path(""),
    plots=True,
    callbacks=Callbacks(),
    compute_loss=None,
    epoch=None,
    rgbt=False,  # RGB-T multispectral input
):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model
        half &= device.type != "cpu"  # half precision only supported on CUDA
        model.half() if half else model.float()
    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # RGB-T 모델 로딩 처리
        if rgbt:
            # RGB-T 모델의 경우 DetectMultiBackend 대신 직접 로딩
            from models.yolo import Model
            
            LOGGER.info("🔥 RGB-T mode: Loading model directly (bypassing DetectMultiBackend)")
            
            # 체크포인트 로드
            weights_path = weights[0] if isinstance(weights, list) else weights
            ckpt = torch.load(weights_path, map_location='cpu')
            
            # 데이터셋 정보 로드
            data_dict = check_dataset(data)
            nc = 1 if single_cls else int(data_dict["nc"])
            
            # 모델 생성
            if 'model' in ckpt and hasattr(ckpt['model'], 'yaml'):
                model_yaml = ckpt['model'].yaml
                LOGGER.info(f"✅ Using model YAML from checkpoint")
            else:
                # 기본 RGB-T 모델 구조 파일 사용
                model_yaml = ROOT / "models/yolov5s_kaist-rgbt.yaml"
                if not model_yaml.exists():
                    LOGGER.warning(f"RGB-T config not found at {model_yaml}, using default yolov5s.yaml")
                    model_yaml = ROOT / "models/yolov5s.yaml"
                LOGGER.info(f"🔧 Using model config: {model_yaml}")
            
            model = Model(model_yaml, ch=3, nc=nc).to(device)
            
            # 가중치 로드
            state_dict = ckpt['model'].float().state_dict() if 'model' in ckpt else ckpt
            model.load_state_dict(state_dict, strict=False)
            
            # 모델 속성 설정 (DetectMultiBackend와 호환성을 위해)
            model.stride = max(int(model.stride.max()), 32)
            model.names = data_dict.get('names', {i: f'class{i}' for i in range(nc)})
            model.pt = True
            
            stride, pt, jit, engine = model.stride, True, False, False
            
            LOGGER.info(f"✅ RGB-T model loaded: {len(state_dict):,} parameters")
            
        else:
            # 기존 DetectMultiBackend 사용 (RGB 전용)
            model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
            stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
            
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        
        # Half precision 설정
        if rgbt:
            half = half and device.type != 'cpu'
            if half:
                model.half()
                LOGGER.info("🔥 RGB-T model using FP16")
        else:
            half = model.fp16  # FP16 supported on limited backends with CUDA
        
        if engine and not rgbt:
            batch_size = model.batch_size
        else:
            if not rgbt and not (pt or jit):
                batch_size = 1  # export.py models default to batch-size 1
                LOGGER.info(f"Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models")

        # Data
        data = check_dataset(data)  # check

    # Configure
    model.eval()
    cuda = device.type != "cpu"
    is_coco = isinstance(data.get("val"), str) and data["val"].endswith(f"coco{os.sep}val2017.txt")  # COCO dataset
    nc = 1 if single_cls else int(data["nc"])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if not training:
        # 클래스 수 검증 (RGB-T가 아닌 경우만)
        if not rgbt and pt and not single_cls:  # check --weights are trained on --data
            ncm = model.model.nc if hasattr(model, 'model') else getattr(model, 'nc', nc)
            assert ncm == nc, (
                f"{weights} ({ncm} classes) trained on different --data than what you passed ({nc} "
                f"classes). Pass correct combination of --weights and --data that are trained together."
            )
        
        # 모델 웜업 처리
        if rgbt:
            # RGB-T 모델 웜업
            LOGGER.info("🔥 Warming up RGB-T model...")
            dummy_rgb = torch.zeros(1 if pt else batch_size, 3, imgsz, imgsz, device=device)
            dummy_thermal = torch.zeros(1 if pt else batch_size, 3, imgsz, imgsz, device=device)
            if half:
                dummy_rgb, dummy_thermal = dummy_rgb.half(), dummy_thermal.half()
            
            try:
                _ = model([dummy_rgb, dummy_thermal])
                LOGGER.info("✅ RGB-T model warmup completed")
            except Exception as e:
                LOGGER.warning(f"⚠️ RGB-T model warmup failed: {e}")
        else:
            # 기존 웜업 방식
            model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # warmup
        
        # 데이터로더 설정
        pad, rect = (0.0, False) if task == "speed" else (0.5, False if rgbt else pt)  # RGB-T에서는 rect=False
        task = task if task in ("train", "val", "test") else "val"  # path to train/val/test images
        
        LOGGER.info(f"📂 Loading {task} dataset: {data[task]}")
        
        dataloader = create_dataloader(
            data[task],
            imgsz,
            batch_size,
            stride,
            single_cls,
            pad=pad,
            rect=rect,
            workers=workers if not rgbt else 0,  # RGB-T에서는 workers=0 권장
            prefix=colorstr(f"{task}: "),
            rgbt_input=rgbt,  # RGB-T 입력 활성화
        )[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = model.names if hasattr(model, "names") else model.module.names  # get class names
    if isinstance(names, (list, tuple)):  # old format
        names = dict(enumerate(names))
    class_map = list(range(1000))
    s = ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "P", "R", "mAP50", "mAP50-95")
    tp, fp, p, r, f1, mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    dt = Profile(device=device), Profile(device=device), Profile(device=device)  # profiling times
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    callbacks.run("on_val_start")
    pbar = tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)  # progress bar

    for batch_i, (ims, targets, paths, shapes, indices) in enumerate(pbar):
        callbacks.run("on_val_batch_start")
        with dt[0]:
            if isinstance(ims, list):
                ims = [im.to(device, non_blocking=True).float() / 255 for im in ims]    # For RGB-T input
                nb, _, height, width = ims[0].shape  # batch size, channels, height, width
                if half:
                    ims = [im.half() for im in ims]
            else:
                ims = ims.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0
                nb, _, height, width = ims.shape  # batch size, channels, height, width
                if half:
                    ims = ims.half()

            targets = targets.to(device)

        # Inference
        with dt[1]:
            preds, train_out = model(ims) if compute_loss else (model(ims, augment=augment), None)

        # Loss
        if compute_loss:
            loss += compute_loss(train_out, targets)[1]  # box, obj, cls

        # NMS
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        with dt[2]:
            preds = non_max_suppression(
                preds, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls, max_det=max_det
            )

        if isinstance(ims, list):
            ims = ims[0]    # thermal image for visualization

        # Metrics
        for si, pred in enumerate(preds):
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            path, shape = Path(paths[si]), shapes[si][0]
            index = indices[si]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            seen += 1

            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                    if plots:
                        confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_boxes(ims[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_boxes(ims[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)

            # Save/log
            if save_txt:
                (save_dir / "labels").mkdir(parents=True, exist_ok=True)
                save_one_txt(predn, save_conf, shape, file=save_dir / "labels" / f"{path.stem}.txt")
            if save_json:
                save_one_json(predn, jdict, path, index, class_map)  # append to COCO-JSON dictionary
            callbacks.run("on_val_image_end", pred, predn, path, names, ims[si])

        # Plot images
        if plots and batch_i < 3:
            desc = f"val_batch{batch_i}" if epoch is None else f"val_epoch{epoch}_batch{batch_i}"
            plot_images(ims, targets, paths, save_dir / f"{desc}_labels.jpg", names)  # labels
            plot_images(ims, output_to_target(preds), paths, save_dir / f"{desc}_pred.jpg", names)  # pred

        callbacks.run("on_val_batch_end", batch_i, ims, targets, paths, shapes, preds)

    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=False, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()

    # filter out ignore labels when counting the number of gt boxes
    lbls = stats[3].astype(int)
    lbls = lbls[lbls >= 0]
    nt = np.bincount(lbls, minlength=nc)  # number of targets per class

    # Print results
    pf = "%22s" + "%11i" * 2 + "%11.3g" * 4  # print format
    LOGGER.info(pf % ("all", seen, nt.sum(), mp, mr, map50, map))
    if nt.sum() == 0:
        LOGGER.warning(f"WARNING ⚠️ no labels found in {task} set, can not compute metrics without labels")

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}" % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        callbacks.run("on_val_end", nt, tp, fp, p, r, f1, ap, ap50, ap_class, confusion_matrix)

    # Save JSON
    if save_json and len(jdict):
        if weights:
            w = Path(weights[0] if isinstance(weights, list) else weights).stem
        else:
            w = f'epoch{epoch}'

        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions
        LOGGER.info(f"\n💾 Saving {pred_json}...")

        with open(pred_json, "w") as f:
            json.dump(jdict, f, indent=2)

        LOGGER.info(f"✅ Saved {len(jdict)} predictions to {pred_json}")

        # KAIST evaluation
        if rgbt:
            LOGGER.info(f"🔍 Evaluating KAIST RGB-T results...")
            try:
                # KAIST annotation 파일 경로들 시도
                kaist_ann_paths = [
                    'datasets/kaist-rgbt/KAIST_val_annotation.json',
                    'utils/eval/KAIST_val-A_annotation.json',
                    'utils/eval/KAIST_test_annotation.json',
                    'KAIST_annotation.json'
                ]
                
                kaist_ann_file = None
                for ann_path in kaist_ann_paths:
                    if os.path.exists(ann_path):
                        kaist_ann_file = ann_path
                        break
                
                if kaist_ann_file:
                    eval_command = f"python utils/eval/kaisteval.py --annFile {kaist_ann_file} --rstFile {pred_json}"
                    LOGGER.info(f"🚀 Running KAIST evaluation: {eval_command}")
                    result = os.system(eval_command)
                    if result == 0:
                        LOGGER.info("✅ KAIST evaluation completed successfully")
                    else:
                        LOGGER.warning("⚠️ KAIST evaluation completed with warnings")
                else:
                    LOGGER.warning("⚠️ KAIST annotation file not found. Please generate it using utils/eval/generate_kaist_ann_json.py")
                    
            except Exception as e:
                LOGGER.info(f"⚠️ KAIST evaluation failed: {e}")

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


def parse_opt():
    """Parses command-line options for YOLOv5 model inference configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="dataset.yaml path")
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt", help="model path(s)")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument("--conf-thres", type=float, default=0.001, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.6, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=300, help="maximum detections per image")
    parser.add_argument("--task", default="val", help="train, val, test, speed or study")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--single-cls", action="store_true", help="treat as single-class dataset")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--verbose", action="store_true", help="report mAP by class")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-hybrid", action="store_true", help="save label+prediction hybrid results to *.txt")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-json", action="store_true", help="save a COCO-JSON results file")
    parser.add_argument("--project", default=ROOT / "runs/val", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--rgbt", action="store_true", help="Feed RGB-T multispectral image pair.")  # RGB-T 모드
    
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    opt.save_json |= opt.data.endswith("coco.yaml")
    opt.save_txt |= opt.save_hybrid
    print_args(vars(opt))
    return opt


def main(opt):
    """Executes YOLOv5 tasks like training, validation, testing, speed, and study benchmarks based on provided
    options.
    """
    # RGB-T 모드 정보 출력
    if opt.rgbt:
        LOGGER.info("🔥" * 20)
        LOGGER.info("🔥 RGB-T MODE ACTIVATED")
        LOGGER.info("🔥" * 20)

    if opt.task in ("train", "val", "test"):  # run normally
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.info(f"WARNING ⚠️ confidence threshold {opt.conf_thres} > 0.001 produces invalid results")
        if opt.save_hybrid:
            LOGGER.info("WARNING ⚠️ --save-hybrid will return high mAP from hybrid labels, not from predictions alone")
        run(**vars(opt))

    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        opt.half = torch.cuda.is_available() and opt.device != "cpu"  # FP16 for fastest results
        if opt.task == "speed":  # speed benchmarks
            # python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt...
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(**vars(opt), plots=False)

        elif opt.task == "study":  # speed vs mAP benchmarks
            # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt...
            for opt.weights in weights:
                f = f"study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt"  # filename to save to
                x, y = list(range(256, 1536 + 128, 128)), []  # x axis (image sizes), y axis
                for opt.imgsz in x:  # img-size
                    LOGGER.info(f"\nRunning {f} --imgsz {opt.imgsz}...")
                    r, _, t = run(**vars(opt), plots=False)
                    y.append(r + t)  # results and times
                np.savetxt(f, y, fmt="%10.4g")  # save
            subprocess.run(["zip", "-r", "study.zip", "study_*.txt"])
            plot_val_study(x=x)  # plot
        else:
            raise NotImplementedError(f'--task {opt.task} not in ("train", "val", "test", "speed", "study")')


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)