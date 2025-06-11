#!/usr/bin/env python3
"""
eval_readings_int.py
--------------------
把 GT / Pred 统一成 5 位整数（舍弃小数点及其后一位）
计算整串全对率 SER 和整数 MAE
"""

import json, csv, argparse
from pathlib import Path
import pandas as pd

def norm_int(s: str) -> str:
    """'9575.9'→'09575'   '272.3'→'00272'"""
    int_part = s.split('.',1)[0]   # 去掉小数部分
    return int_part.zfill(5)[:5]   # 补零到 5 位

def load_pred(json_path):
    raw = json.loads(Path(json_path).read_text())
    return {fn: norm_int(v) for fn, v in raw.items()}

def load_gt(csv_path):
    gt={}
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            fn  = Path(row['filename']).with_suffix('.png').name
            gt[fn] = norm_int(row['number'])
    return gt

def main(args):
    pred = load_pred(args.pred)
    gt   = load_gt(args.gt)

    rows, correct, abs_errs = [], 0, []
    for fn, gtnum in gt.items():
        if fn not in pred:
            print('[MISS pred]', fn); continue
        pnum = pred[fn]
        is_ok= (pnum == gtnum)
        correct += is_ok
        abs_errs.append(abs(int(pnum) - int(gtnum)))
        rows.append([fn, gtnum, pnum, is_ok])

    ser = correct/len(rows)
    mae = sum(abs_errs)/len(rows)

    print(f'SER(5位整数全对率): {ser:.3f} ({correct}/{len(rows)})')
    print(f'整数 MAE          : {mae:.2f}')

    df = pd.DataFrame(rows, columns=['filename','gt_int','pred_int','correct'])
    df['abs_diff'] = abs_errs
    df.to_csv('eval_report_int.csv', index=False)
    print('详细结果写入 eval_report_int.csv')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True, help='readings.json')
    ap.add_argument('--gt',   required=True, help='label.csv')
    main(ap.parse_args())
