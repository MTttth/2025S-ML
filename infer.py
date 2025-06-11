#!/usr/bin/env python3
"""
infer_digits_join.py
--------------------
读取 out_digits/ 中 hefei_*_0-5.png
用 classifier_best.pth 逐片分类 → 拼串 → 写 JSON

用法:
python infer_digits_join.py \
       --crop_dir out_digits \
       --model    models/classifier_best.pth \
       --out      readings.json
"""

import torch, argparse, json, cv2
from pathlib import Path
from collections import defaultdict
from torchvision import transforms
from PIL import Image

# ---------------- 复用训练脚本里的 transform ----------------
def build_transform(train=False):
    aug=[]
    if train:
        aug += [
            transforms.RandomAffine(degrees=5, translate=(0.05,0.05)),
            transforms.ColorJitter(0.3,0.3,0.2,0.05)
        ]
    aug += [
        transforms.Grayscale(),
        transforms.Resize((28,28)),
        transforms.ToTensor(),
        transforms.Normalize(0.5,0.5)
    ]
    return transforms.Compose(aug)

tf_infer = build_transform(False)

# ---------------- SimpleCNN 与训练一致 ----------------
import torch.nn as nn
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
    def forward(self,x): return self.net(x)

def load_model(pth, device):
    model = SimpleCNN().to(device)
    sd    = torch.load(pth, map_location=device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model

# ---------------- 单张数字切片推理 ----------------
def predict_digit(img_bgr, model, device):
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    x   = tf_infer(pil).unsqueeze(0).to(device)      # [1,1,28,28]
    with torch.no_grad():
        pred = model(x).argmax(1).item()
    return str(pred)                                 # '0'..'9'

# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--crop_dir', required=True, help='out_digits 文件夹')
    ap.add_argument('--model',    required=True, help='classifier_best.pth')
    ap.add_argument('--out',      default='readings.json')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = load_model(args.model, device)

    crop_dir = Path(args.crop_dir)
    groups   = defaultdict(dict)      # {hefei_3188: {0:path,1:path,…}}

    for p in crop_dir.glob('*_[0-5].png'):
        stem, idx = p.stem.rsplit('_',1)
        groups[stem][int(idx)] = p

    results = {}
    for stem, mp in groups.items():
        digits=[]
        for i in range(6):
            if i not in mp:           # 缺图 → 补 0
                digits.append('0')
                continue
            img = cv2.imread(str(mp[i]))
            digits.append(predict_digit(img, model, device))
        reading = ''.join(digits[:5]) + '.' + digits[5]
        results[stem+'.png'] = reading
        print(f'{stem}.png -> {reading}')

    Path(args.out).write_text(json.dumps(results, indent=2))
    print('Saved to', args.out)

if __name__ == '__main__':
    main()
