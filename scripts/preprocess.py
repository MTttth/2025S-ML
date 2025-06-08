"""
对原始电表照片做去光照、去噪、（可选）畸变校正和透视校正，得到更干净、水平的灰度图或 BGR 图，以供后续检测／分割使用。
输入：../data/raw_images/*.jpg
输出：../data/preprocessed/*.jpg
"""
import cv2
import numpy as np
import glob
import os

def order_points(pts):
    """
    将任意顺序的四个点排序为：左上、右上、右下、左下。
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # 左上最小和
    rect[2] = pts[np.argmax(s)]  # 右下最大和

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 右上最小差
    rect[3] = pts[np.argmax(diff)]  # 左下最大差
    return rect

def remove_light(img):
    """
    对 BGR 图做灰度转换 + 自适应直方图均衡化（CLAHE） + 双边滤波去噪，返回灰度图。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_clahe = clahe.apply(gray)
    denoised = cv2.bilateralFilter(gray_clahe, 5, 75, 75)
    return denoised

def undistort(img, K, distCoeffs):
    """
    如果已有相机内参 K 和畸变系数 distCoeffs，则做畸变校正。
    返回校正后的图像。
    """
    return cv2.undistort(img, K, distCoeffs)

def four_point_transform(img, pts):
    """
    给定 4 个角点（任意顺序），自动排序并对图像做透视变换，返回正视后的矩形图。
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # 计算目标尺寸
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    return warped

if __name__ == '__main__':
    src_dir = '../data/raw_images'
    dst_dir = '../data/preprocessed'
    os.makedirs(dst_dir, exist_ok=True)

    # 可选：定义相机内参和畸变系数以启用畸变校正
    # K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    # distCoeffs = np.array([k1, k2, p1, p2, k3])

    # 可选：透视校正角点示例（手工或检测获得）
    # example_pts = np.array([[x1,y1], [x2,y2], [x3,y3], [x4,y4]], dtype="float32")

    for img_path in glob.glob(os.path.join(src_dir, '*.jpg')):
        img = cv2.imread(img_path)
        # 1. 去光照、去噪，得到灰度图
        clean_gray = remove_light(img)

        # 2. （可选）畸变校正：
        # clean_gray = undistort(clean_gray, K, distCoeffs)

        # 3. 将灰度转回 BGR（如后续需要三通道）
        clean_bgr = cv2.cvtColor(clean_gray, cv2.COLOR_GRAY2BGR)

        # 4. （可选）透视校正：
        # warped = four_point_transform(clean_bgr, example_pts)
        # output = warped
        output = clean_bgr

        # 5. 保存预处理结果
        filename = os.path.basename(img_path)
        save_path = os.path.join(dst_dir, filename)
        cv2.imwrite(save_path, output)
        print(f'[INFO] Saved preprocessed image to {save_path}')
