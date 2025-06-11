#!/usr/bin/env python3
"""
检测数字区并裁剪 ROI
--------------------
输入 : data/preprocessed/*.jpg  (或 --src 指定)
输出 : data/regions/{filename}.jpg
"""
from pathlib import Path
from ultralytics import YOLO
import cv2, argparse, json
from tqdm import tqdm

def get_latest_weight():
    exps = sorted(
        Path("runs/detector").glob("exp*"),
        key=lambda p: p.stat().st_mtime
    )
    exps = [p for p in exps if (p / "weights/best.pt").exists()]
    if not exps:
        raise FileNotFoundError("未在 runs/detector 找到任何权重！")
    return exps[-1] / "weights/best.pt"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(get_latest_weight()))
    ap.add_argument("--src", default="data/preprocessed")
    ap.add_argument("--out", default="data/regions")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    model = YOLO(args.weights)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_log = []
    for img_path in tqdm(list(Path(args.src).glob("*.jpg")), desc="detect"):
        res = model(img_path, conf=args.conf, iou=0.4, verbose=False)[0]
        if not res.boxes:
            print(f"[WARN] no box {img_path.name}")
            continue
        # 取最高分框
        b = res.boxes.xyxy[res.boxes.conf.argmax()].cpu().numpy().astype(int)
        x1,y1,x2,y2 = b
        img = cv2.imread(str(img_path))
        roi = img[y1:y2, x1:x2]
        save_path = out_dir / img_path.name
        cv2.imwrite(str(save_path), roi)
        pred_log.append({"filename": img_path.name, "bbox": [int(i) for i in b]})

    (out_dir / "regions.json").write_text(json.dumps(pred_log, indent=2))
    print(f"[INFO] saved to {out_dir}")


if __name__ == "__main__":
    main()