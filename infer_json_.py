#!/usr/bin/env python3
"""
infer_digits_join.py
--------------------
从裁剪目录 out_digits/ 读取 hefei_*_0-5.png
用 classifier_best.pth (SimpleCNN) 识别 -> 拼整串
"""

import torch, cv2, json, argparse
import torch.nn as nn
from pathlib import Path
from collections import defaultdict
from torchvision import transforms
import numpy as np

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
            nn.Linear(64*7*7, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, nc)
        )
    def forward(self, x): return self.net(x)


# ---------- 预处理 Pipeline ----------
preprocess = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize(0.5,0.5)     # (x-0.5)/0.5  得到 [-1,1] 区间
])

def load_img_tensor(path, device):
    img = cv2.imread(str(path))
    # cv2 返回 BGR; 直接给 torchvision.Grayscale() (会先转 PIL)更简:
    from PIL import Image
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return preprocess(pil).unsqueeze(0).to(device)   # [1,1,28,28]

def main(args):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN().to(dev)
    model.load_state_dict(torch.load(args.model, map_location=dev))
    model.eval()

    crop_dir = Path(args.crop_dir)
    files = defaultdict(dict)      # key=hefei_xxx  value={idx:path}
    for p in crop_dir.glob('*_[0-5].png'):
        stem, idx = p.stem.rsplit('_',1)
        files[stem][int(idx)] = p

    out = {}
    for stem, fmap in files.items():
        digits = []
        for i in range(6):
            path = fmap.get(i)
            if path is None:
                digits.append('0')
                continue
            inp = load_img_tensor(path, dev)
            with torch.no_grad():
                pred = model(inp).argmax(1).item()
            digits.append(str(pred))
        number = ''.join(digits[:5]) + '.' + digits[5]
        out[stem+'.png'] = number
        print(f'{stem}.png -> {number}')

    Path(args.out).write_text(json.dumps(out, indent=2))
    print('Saved to', args.out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--crop_dir', required=True, help='out_digits 目录')
    ap.add_argument('--model',    required=True, help='classifier_best.pth')
    ap.add_argument('--out',      default='readings.json')
    args = ap.parse_args()
    main(args)
