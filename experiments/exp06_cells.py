"""実験06: 200 m の格子ごとに「飛行機の成分・車の成分」を当てる — 見たことのない場所で確率を出す

着想: 持ち主の案（2026-11-01）「飛行機の墓場で、飛行機ありの格子と道路・建物などその他の格子から飛行機の成分を学び、車の処分場から車の成分。
      最終的に飛行場を中心に 10 km を 200 m の格子で、飛行機の成分や車の成分を見る」。
方法:
    データ  tools/build_cell_chips.py の格子の切り出し（Sentinel-1 RTC の VV・VH、24 × 24 画素 = 240 m、dB）。日付ごとに 1 個
            正解は NAIP の航空写真で Claude が目で印をつけた格子（2 = 飛行機あり、1 = 車あり、0 = その他）
    分け方  訓練 = ロズウェル（飛行機）+ ロングビーチ（車）の古い日付、テスト = **ビクタービル（訓練で見ていない飛行機の場所）** + ロングビーチの新しい 3 日付
    条件:
      feat     手で作った特徴（VV・VH の平均・ばらつき・上位 10%・明るい画素の割合）+ ロジスティック回帰
      cnn      小さな CNN（VV・VH の 2 チャネル）
      sin_amp  Sin の板の光路（VV の振幅を光の場に、32 × 32 の真ん中に置く）
    学習: 種類の数の違いをならす重み、3000 steps、seed 3
指標: ビクタービルで「飛行機あり」の確率の当たり（AUC = 飛行機ありの格子を、その他より高く並べられる割合）、ロングビーチの新しい日付で「車あり」の AUC、
      日付を平均した確率の地図（図は tools/fig_exp06.py）
結果: results/exp06.txt（1 行 JSON）、確率は cache/exp06_<条件>_<seed>.pt
実行: CUDA_VISIBLE_DEVICES= .venv/Scripts/python experiments/exp06_cells.py --model cnn --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
import vendor.sn2_parts_2d_v0_11_0 as p2  # noqa: E402

from zuikoh.config import data_dir  # noqa: E402

L = 32


def auc(score, pos):
    """順位で求める AUC（正の格子を負の格子より高く並べられる割合）。"""
    s, p = np.asarray(score), np.asarray(pos, bool)
    if p.sum() == 0 or (~p).sum() == 0:
        return float("nan")
    r = s.argsort().argsort() + 1
    return float((r[p].sum() - p.sum() * (p.sum() + 1) / 2) / (p.sum() * (~p).sum()))


def features(X):
    """手で作った特徴: VV・VH の平均・ばらつき・上位 10%・−5 dB より明るい画素の割合。"""
    f = []
    for c in range(2):
        v = X[:, c].flatten(1)
        f += [v.mean(1), v.std(1), v.quantile(0.9, dim=1), (v > -5).float().mean(1)]
    return torch.stack(f, 1)


class SmallCNN(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        self.c1 = nn.Conv2d(2, width, 3, padding=1)
        self.c2 = nn.Conv2d(width, width, 3, padding=1)
        self.fc = nn.Linear(width, 3)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.relu(self.c2(x))
        return self.fc(x.mean((-2, -1)))


def to_field(X):
    """VV の dB → 振幅（いちばん明るい画素で割る）→ 32 × 32 の場の真ん中に置く（光の場）。"""
    a = 10 ** (X[:, 0] / 20)
    a = a / (a.amax((-2, -1), keepdim=True) + 1e-12)
    o = (L - a.shape[-1]) // 2
    f = torch.zeros(len(a), L, L)
    f[:, o:o + a.shape[-2], o:o + a.shape[-1]] = a
    return f.to(torch.complex64)


def main():
    ap = argparse.ArgumentParser(description="実験06: 格子ごとの飛行機・車の成分")
    ap.add_argument("--model", required=True, choices=["feat", "cnn", "sin_amp"])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--plates", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "exp06.txt"))
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    torch.set_num_threads(4)
    D = torch.load(data_dir() / "processed" / "cell_chips" / "cells.pt", weights_only=False)
    X, y = D["X"], D["y"]
    lb_dates = sorted({d for d, s in zip(D["date"], D["site"]) if s == "longbeach_cars"})
    lb_test = set(lb_dates[-3:])
    te = torch.tensor([s == "victorville" or (s == "longbeach_cars" and d in lb_test) for s, d in zip(D["site"], D["date"])])
    Xtr, ytr, Xte, yte = X[~te], y[~te], X[te], y[te]
    mu = Xtr.mean((0, 2, 3), keepdim=True)
    sd = Xtr.std((0, 2, 3), keepdim=True)
    w = torch.bincount(ytr, minlength=3).float()
    w = (w.sum() / (3 * w.clamp(min=1)))
    t0 = time.time()
    g = torch.Generator().manual_seed(a.seed)
    if a.model == "feat":
        ftr, fte = features(Xtr), features(Xte)
        m_, s_ = ftr.mean(0), ftr.std(0) + 1e-6
        net = nn.Linear(ftr.shape[1], 3)
        opt = torch.optim.Adam(net.parameters(), lr=0.05)
        for _ in range(2000):
            loss = F.cross_entropy(net((ftr - m_) / s_), ytr, weight=w)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            prob = F.softmax(net((fte - m_) / s_), 1)
    else:
        net = SmallCNN() if a.model == "cnn" else p2.Net2D(L, 3, plates=("stripes",) * a.plates, patches=8, z=32.0, seed=a.seed, window=3)
        net = net.to(a.device)
        opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=0.01)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
        prep = (lambda x: (x - mu) / sd) if a.model == "cnn" else to_field
        for _ in range(a.steps):
            idx = torch.randint(len(Xtr), (256,), generator=g)
            xb = Xtr[idx]
            if torch.rand(1, generator=g).item() < 0.5:     # 左右・上下の反転で水増し（格子の中の向きは決まっていない）
                xb = xb.flip(-1)
            if torch.rand(1, generator=g).item() < 0.5:
                xb = xb.flip(-2)
            loss = F.cross_entropy(net(prep(xb).to(a.device)), ytr[idx].to(a.device), weight=w.to(a.device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
        with torch.no_grad():
            prob = torch.cat([F.softmax(net(prep(Xte[i:i + 1024]).to(a.device)), 1).cpu() for i in range(0, len(Xte), 1024)])
    params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    site_te = np.array([s for s, t in zip(D["site"], te.tolist()) if t])
    v = site_te == "victorville"
    lbm = site_te == "longbeach_cars"
    yt = yte.numpy()
    pr = prob.numpy()
    rec = dict(vars(a), exp="06", params=int(params), train_s=round(time.time() - t0, 1), torch=torch.__version__,
               version="vendor sn2_parts_2d 0.11.0", region="roswell+longbeach_cars → victorville",
               n_train=int(len(ytr)), n_test_vcv=int(v.sum()), n_test_lb=int(lbm.sum()),
               auc_aircraft_vcv=round(auc(pr[v, 2], yt[v] == 2), 4), auc_car_lb=round(auc(pr[lbm, 1], yt[lbm] == 1), 4),
               aircraft_prob_mean_pos_vcv=round(float(pr[v & (yt == 2), 2].mean()), 3), aircraft_prob_mean_neg_vcv=round(float(pr[v & (yt != 2), 2].mean()), 3))
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    torch.save(dict(prob=prob, site=site_te.tolist(), y=yte, cx=D["cx"][te], cy=D["cy"][te],
                    date=[d for d, t in zip(D["date"], te.tolist()) if t]), data_dir() / "cache" / f"exp06_{a.model}_{a.seed}.pt")
    print(json.dumps({k: rec[k] for k in ["model", "seed", "params", "train_s", "auc_aircraft_vcv", "auc_car_lb",
                                         "aircraft_prob_mean_pos_vcv", "aircraft_prob_mean_neg_vcv"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
