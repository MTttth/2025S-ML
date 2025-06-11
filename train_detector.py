#!/usr/bin/env python3
"""
训练目标检测模型 (YOLOv8)
---------------------------------
1. 把 Pascal-VOC XML 转成 YOLO txt
2. 生成 data.yaml
3. 调 ultralytics YOLO 训练
"""
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from tqdm import tqdm
import argparse
import yaml
from ultralytics import YOLO

DEF_LABEL = "digit_region"   # 所有标注都是一个类

def voc2yolo(xml_path, img_wh):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    w, h = img_wh
    txt_lines = []
    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        xmin = int(float(bbox.find("xmin").text))
        ymin = int(float(bbox.find("ymin").text))
        xmax = int(float(bbox.find("xmax").text))
        ymax = int(float(bbox.find("ymax").text))
        # 转归一化中心坐标
        cx = (xmin + xmax) / 2 / w
        cy = (ymin + ymax) / 2 / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h
        txt_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return "\n".join(txt_lines)

def convert_split(split):
    img_dir = Path(f"data/meter_dataset/images/{split}")
    ann_dir = Path(f"data/meter_dataset/annotations/{split}")
    yolo_lbl_dir = Path(f"data/meter_dataset/labels/{split}")
    yolo_lbl_dir.mkdir(parents=True, exist_ok=True)
    for img_path in tqdm(list(img_dir.glob("*.jpg")), desc=f"XML→YOLO ({split})"):
        xml_path = ann_dir / f"{img_path.stem}.xml"
        if not xml_path.exists():
            continue
        # 读取分辨率
        import cv2
        img = cv2.imread(str(img_path))
        txt = voc2yolo(xml_path, (img.shape[1], img.shape[0]))
        (yolo_lbl_dir / f"{img_path.stem}.txt").write_text(txt)

def prepare_yaml():
    data = {
        "path": "data/meter_dataset",
        "train": "images/train",
        "val": "images/val",
        "names": {0: DEF_LABEL}
    }
    Path("data/meter_dataset/data.yaml").write_text(yaml.dump(data))


def main():
    default_dev = "cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--device", default=default_dev)
    args = ap.parse_args()

    convert_split("train")
    convert_split("val")
    prepare_yaml()

    model = YOLO(args.model)
    model.train(data="data/meter_dataset/data.yaml",
                epochs=args.epochs,
                imgsz=args.imgsz,
                device=args.device,
                project="runs/detector",
                name="exp",
                patience=20)


if __name__ == "__main__":
    main()