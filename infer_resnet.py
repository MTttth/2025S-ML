#!/usr/bin/env python3
"""
infer_resnet.py
---------------
用 resnet18_digits.pth 识别 out_digitsV2 中切片,
拼成完整读数并写入 JSON
"""

import torch, argparse, json, cv2
from pathlib import Path
from collections import defaultdict
from torchvision import transforms
import timm
from PIL import Image

# ---------------- transform (与训练脚本保持一致) -------------
tf = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(0.5,0.5)
])

def load_model(pth, device):
    net = timm.create_model('resnet18', pretrained=False,
                            in_chans=1, num_classes=10)
    net.load_state_dict(torch.load(pth, map_location=device), strict=True)
    net.eval().to(device)
    return net

def predict_digit(img_bgr, net, device):
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    x   = tf(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        return str(net(x).argmax(1).item())

def main(a):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = load_model(a.model, dev)

    groups = defaultdict(dict)
    for p in Path(a.crop_dir).glob('*_[0-5].png'):
        stem, idx = p.stem.rsplit('_',1)
        groups[stem][int(idx)] = p

    res={}
    for stem, mp in groups.items():
        digits=[]
        for i in range(6):
            if i not in mp:
                digits.append('0')
                continue
            img = cv2.imread(str(mp[i]))
            digits.append(predict_digit(img, net, dev))
        res[stem+'.png'] = ''.join(digits[:5])+'.'+digits[5]   # 根据你的文件扩展名调整
        print(stem, '->', res[stem+'.png'])

    Path(a.out).write_text(json.dumps(res, indent=2))
    print('saved to', a.out)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--crop_dir', required=True)  # ./out_digitsV2
    ap.add_argument('--model',    required=True)  # resnet18_digits.pth
    ap.add_argument('--out',      default='readings.json')
    main(ap.parse_args())
