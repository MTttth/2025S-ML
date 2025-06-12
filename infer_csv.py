#!/usr/bin/env python3
"""
infer_digits_join_csv.py
------------------------
从裁剪目录 out_digits/ 读取 hefei_*_0-5.png
用 classifier_best.pth 识别 → 拼整串
输出 CSV: id,reading
"""

import torch, cv2, argparse, csv
from pathlib import Path
from collections import defaultdict
from torchvision import transforms
from PIL import Image
import torch.nn as nn

# ---------- SimpleCNN (同训练脚本) ----------
class SimpleCNN(nn.Module):
    def __init__(self, nc=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.ReLU(),
            nn.Conv2d(32,32,3,padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*7*7,256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, nc)
        )
    def forward(self,x): return self.net(x)

# ---------- 预处理 Pipeline ----------
preprocess = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize(0.5,0.5)
])

def load_img_tensor(path, device):
    img = cv2.imread(str(path))
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return preprocess(pil).unsqueeze(0).to(device)

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--crop_dir', required=True, help='out_digits 目录')
    ap.add_argument('--model',    required=True, help='classifier_best.pth')
    ap.add_argument('--out',      default='readings.csv', help='输出 CSV 文件')
    args = ap.parse_args()

    dev   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN().to(dev)
    model.load_state_dict(torch.load(args.model, map_location=dev))
    model.eval()

    # 收集切片
    groups = defaultdict(dict)
    for p in Path(args.crop_dir).glob('*_[0-5].png'):
        stem, idx = p.stem.rsplit('_',1)
        groups[stem][int(idx)] = p

    # 写 CSV
    with open(args.out, 'w', newline='') as f:
        writer = csv.writer(f)
        # writer.writerow(['id','reading'])  # 如需表头可取消注释
        for stem, fmap in sorted(groups.items()):
            digits=[]
            for i in range(6):
                path = fmap.get(i)
                if path is None:
                    digits.append('0')
                else:
                    inp = load_img_tensor(path, dev)
                    with torch.no_grad():
                        d = model(inp).argmax(1).item()
                    digits.append(str(d))
            reading = ''.join(digits[:5]) + '.' + digits[5]
            fname = stem
            writer.writerow([fname, reading])
            print(f'{fname} -> {reading}')

    print(f'Saved CSV to {args.out}')

if __name__ == '__main__':
    main()
