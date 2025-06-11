#!/usr/bin/env python3
"""
crop_six.py
-----------
将 /data/regions/*.jpg 按 YOLOv8 检测框裁成 6 张数字小图
模型: /runs/detect/train4/weights/best.pt
输出: out_digits/{原名}_{idx}.png   (idx 0‥5 按 x 坐标排序)
"""

from ultralytics import YOLO
import cv2, numpy as np
from pathlib import Path
from tqdm import tqdm

IMG_DIR = Path('./data/regions')
WEIGHTS = './runs/detect/train6/best.pt'
OUTDIR  = Path('out_digitsV2')
OUTDIR.mkdir(exist_ok=True)

model = YOLO(WEIGHTS)

def get_boxes(img, conf=0.25):
    res = model(img, conf=conf, iou=0.4, verbose=False)[0]
    if res.boxes is None: return []
    return [box.int().tolist() for box in res.boxes.xyxy]

def crop_one(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        print('[err] read fail', image_path); return
    h,w,_ = img.shape
    boxes = get_boxes(img)
    print(f'Found {len(boxes)} boxes in {image_path.name}')
    if len(boxes) < 6:                       # 置信度过高 → 降一下再跑
        boxes = get_boxes(img, conf=0.1)
    # 只取最靠左的 6 个
    boxes = sorted(boxes, key=lambda b: b[0])[:6]
    # 若>6 也只取 6
    for idx,b in enumerate(boxes):
        x1,y1,x2,y2 = b
        crop = img[y1:y2, x1:x2]
        if crop.size==0: continue
        cv2.imwrite(str(OUTDIR/f'{image_path.stem}_{idx}.png'), crop)

if __name__ == '__main__':
    pics = list(IMG_DIR.glob('*.jpg')) + list(IMG_DIR.glob('*.png'))
    if not pics:
        print('No images found in', IMG_DIR); exit()
    for p in tqdm(pics, desc='Cropping'):
        crop_one(p)
    print('Done. Crops saved to', OUTDIR)
