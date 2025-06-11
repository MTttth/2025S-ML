#!/usr/bin/env python3
"""
预处理：去光照、去噪、透视校正
--------------------------------
输入 : data/raw_images/*.jpg
输出 : data/preprocessed/*.jpg
"""
import cv2
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
from skimage import exposure

def deskew(img, max_skew=10):
    """利用霍夫线估计透视/旋转角度，做轻量 deskew。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    if lines is None:
        return img
    angles = []
    for l in lines[:20]:
        rho, theta = l[0]
        angle = (theta - np.pi/2) * 180/np.pi
        if abs(angle) < max_skew:
            angles.append(angle)
    if not angles:
        return img
    median_angle = np.median(angles)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), median_angle, 1)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def deskew_v2(img, min_area_thresh=5000):
    """
    更稳健的 deskew：
    1. Canny + 形态学 → 最大轮廓
    2. minAreaRect 得到旋转角度
    3. warpAffine 旋转整图
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 1. 边缘 & 闭运算，连通数字区域
    edges = cv2.Canny(gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 2. 找最大轮廓
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return img
    # 过滤小面积噪声
    cnts = [c for c in cnts if cv2.contourArea(c) > min_area_thresh]
    if not cnts:
        return img

    main_cnt = max(cnts, key=cv2.contourArea)

    # 3. 用最小外接矩形拟合主轮廓
    rect = cv2.minAreaRect(main_cnt)
    ((cx, cy), (w, h), angle) = rect
    # cv2.minAreaRect 的 angle 定义有时是 [-90,-0) 或 [0,90)，根据 w>h 调整
    if w < h:
        angle = angle + 90

    # 4. 旋转整图
    M = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), angle, 1)
    deskewed = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    return deskewed

def deskew_v3(img, max_skew=15, min_line_len=100, max_line_gap=10):
    """
    基于 HoughLinesP 的轻量 deskew：
    1. Canny 提取边缘
    2. HoughLinesP 找线段，只保留接近水平、足够长的
    3. 计算这些线段的角度中位数 median_angle
    4. 以 -median_angle 旋转整图（不会翻转）
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # P 型 Hough：输出每条线段的端点
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=100,
        minLineLength=min_line_len,
        maxLineGap=max_line_gap
    )
    if lines is None:
        return img

    angles = []
    for x1,y1,x2,y2 in lines[:,0]:
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            continue
        angle = np.degrees(np.arctan2(dy, dx))
        # 只保留小于 max_skew 绝对值的角度（即近水平）
        if abs(angle) < max_skew:
            angles.append(angle)

    if not angles:
        return img

    # 中位数更鲁棒
    median_angle = np.median(angles)

    # 构造旋转矩阵，注意符号：向右倾（正角度）需要逆时针旋转（负角度）来矫正
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), -median_angle, 1.0)
    deskewed = cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )
    return deskewed

def enhance(img):
    """CLAHE + 亮度归一化."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def process_one(path_in, path_out):
    img = cv2.imread(str(path_in))
    if img is None:
        print(f"[WARN] cannot read {path_in}")
        return
    img = deskew_v3(img)
    img = enhance(img)
    # Gamma & denoise
    # img = exposure.adjust_gamma(img, 1.2)
    # img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    cv2.imwrite(str(path_out), img)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="data/raw_images", help="原始图片目录")
    ap.add_argument("--out_dir", default="data/preprocessed", help="输出目录")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = list(in_dir.glob("*.jpg"))
    for p in tqdm(images, desc="preprocess"):
        process_one(p, out_dir / p.name)


if __name__ == "__main__":
    main()