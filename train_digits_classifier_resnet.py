import torch, argparse
from pathlib import Path
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm, numpy as np
from tqdm import tqdm

def tf(train):
    aug=[]
    if train:
        aug += [transforms.RandomAffine(8, (0.08,0.08)),
                transforms.ColorJitter(0.4,0.4,0.3,0.1)]
    aug += [transforms.Grayscale(),
            transforms.Resize((64,64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])]
    return transforms.Compose(aug)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--tr',default='data/digits/train')
    ap.add_argument('--val',default='data/digits/val')
    ap.add_argument('--epochs',type=int,default=40)
    ap.add_argument('--bs',type=int,default=64)
    ap.add_argument('--out',default='models/resnet18_digits.pth')
    args=ap.parse_args()

    train_ds = datasets.ImageFolder(args.tr, tf(True))
    val_ds   = datasets.ImageFolder(args.val, tf(False))
    tr_ld = DataLoader(train_ds, args.bs, True ,num_workers=4)
    va_ld = DataLoader(val_ds,   args.bs, False,num_workers=2)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = timm.create_model('resnet18', pretrained=False,
                              in_chans=1, num_classes=len(train_ds.classes))
    model.to(dev)

    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    opt  = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    best = 0
    for ep in range(args.epochs):
        model.train(); tot=0
        for x,y in tqdm(tr_ld,f'E{ep} train'):
            x,y = x.to(dev),y.to(dev)
            opt.zero_grad(); loss=crit(model(x),y); loss.backward(); opt.step()
            tot += loss.item()*y.size(0)
        # ---- val
        model.eval(); corr=totv=0
        with torch.no_grad():
            for x,y in tqdm(va_ld,leave=False):
                x,y=x.to(dev),y.to(dev)
                corr += (model(x).argmax(1)==y).sum().item()
                totv += y.size(0)
        acc = corr/totv
        print(f'E{ep}: loss={tot/len(train_ds):.3f}  valAcc={acc:.3f}')
        if acc>best:
            best=acc; Path(args.out).parent.mkdir(parents=True,exist_ok=True)
            torch.save(model.state_dict(),args.out)
            print('  save best', best)

if __name__=='__main__': main()
