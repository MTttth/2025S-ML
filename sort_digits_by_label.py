#!/usr/bin/env python3
"""
sort_digits_by_label.py
-----------------------
根据 label.csv 中的真值，把切好的 hefei_XXXX_0-5.jpg
移动到 digits/train/0~9 子文件夹。

用法：
    python sort_digits_by_label.py \
        --csv   label.csv \
        --crop  crops \
        --out   digits/train \
        --move              # 默认复制，带 --move 则移动
"""

import csv, shutil, argparse
from pathlib import Path

def parse_number(num_str):
    """去掉小数点并左侧补 0 → 返回 6 位字符串"""
    num = num_str.replace('.', '')
    return num.zfill(6)

def main(args):
    csv_path = Path(args.csv)
    crop_dir = Path(args.crop)
    out_root = Path(args.out)

    # 创建 0–9 子目录
    for d in range(10):
        (out_root / str(d)).mkdir(parents=True, exist_ok=True)

    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            stem = Path(row['filename']).stem        # hefei_3188
            digits = parse_number(row['number'])     # '095759'
            if len(digits) != 6:
                print(f"[skip] {stem}: label 长度异常 -> {digits}")
                continue

            for idx, char in enumerate(digits):
                crop_name = f"{stem}_{idx}.png"
                src = crop_dir / crop_name
                if not src.exists():
                    print(f"[warn] 缺少 {src}")
                    continue
                dst = out_root / char / crop_name
                if args.move:
                    shutil.move(src, dst)
                else:
                    shutil.copy(src, dst)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv',  required=True, help='label.csv 路径')
    ap.add_argument('--crop', required=True, help='hefei_xxxx_0-5.jpg 所在目录')
    ap.add_argument('--out',  required=True, help='输出根目录 digits/train')
    ap.add_argument('--move', action='store_true', help='移动而非复制')
    args = ap.parse_args()
    main(args)
