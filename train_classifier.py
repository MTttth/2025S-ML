#!/usr/bin/env python3
"""
改进数字分类器训练：
- 与推理共享 transform
- 数据增强：随机亮度、仿射
- BatchBalancedSampler 保持各类均衡
- CosineLR 调度
"""
import torch, argparse, random
from pathlib import Path
from torch import nn, optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from tqdm import tqdm

charset = "0123456789"   # dot 如果要识别，加在最后

def build_transform(train=True):
    aug = []
    if train:
        aug += [
            transforms.RandomAffine(degrees=5, translate=(0.05,0.05)),
            transforms.ColorJitter(0.3,0.3,0.2,0.05)
        ]
    aug += [
        transforms.Grayscale(),
        transforms.Resize((28,28)),
        transforms.ToTensor(),                 # 0–1
        transforms.Normalize(0.5,0.5)          # (-.5)/.5 → [-1,1]
    ]
    return transforms.Compose(aug)

class SimpleCNN(nn.Module):
    def __init__(self, nc):
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

def make_balanced_sampler(dataset):
    counts = [0]*len(dataset.classes)
    for _,y in dataset:
        counts[y]+=1
    weights = [1.0/counts[y] for _,y in dataset]
    return WeightedRandomSampler(weights, len(dataset))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--train','-tr',default='data/digits/train')
    ap.add_argument('--val','-val', default='data/digits/val')
    ap.add_argument('--epochs',type=int,default=100)
    ap.add_argument('--bs',type=int,default=128)
    ap.add_argument('--lr',type=float,default=1e-3)
    ap.add_argument('--out',default='models/classifier_best.pth')
    args=ap.parse_args()

    train_ds = datasets.ImageFolder(args.train, build_transform(True))
    val_ds   = datasets.ImageFolder(args.val,   build_transform(False))

    sampler  = make_balanced_sampler(train_ds)
    tr_ld = DataLoader(train_ds, batch_size=args.bs, sampler=sampler, num_workers=4)
    va_ld = DataLoader(val_ds,   batch_size=args.bs, shuffle=False, num_workers=2)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = SimpleCNN(len(train_ds.classes)).to(dev)
    crit= nn.CrossEntropyLoss()
    opt = optim.Adam(net.parameters(), lr=args.lr)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best=0
    for ep in range(args.epochs):
        net.train(); tl=0
        for x,y in tqdm(tr_ld,desc=f'E{ep}'):
            x,y = x.to(dev),y.to(dev)
            opt.zero_grad()
            loss = crit(net(x),y); loss.backward(); opt.step()
            tl += loss.item()*y.size(0)
        sch.step()
        # —— 验证
        net.eval(); corr=tot=0
        with torch.no_grad():
            for x,y in va_ld:
                x,y = x.to(dev),y.to(dev)
                pred=net(x).argmax(1)
                corr += (pred==y).sum().item()
                tot  += y.size(0)
        acc = corr/tot
        print(f'E{ep} trainLoss={tl/len(train_ds):.3f}  valAcc={acc:.3f}')
        if acc>best:
            best=acc; Path(args.out).parent.mkdir(exist_ok=True,parents=True)
            torch.save(net.state_dict(), args.out)
            print('  ↳ saved best', best)

if __name__=='__main__':
    main()