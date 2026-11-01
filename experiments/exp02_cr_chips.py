"""実験02: 角の反射板を見分ける — Sin の板（複素数の光の場）と比較相手、反射板ごとに訓練とテストを分ける

着想: 持ち主の依頼（2026-10-31 夜）「複数の角度や違う日の同じ点、オクラホマの同じ形の反射の点を、訓練とテストに分けて学習」。
      est_02 で、反射板は孤立したきれいな点として写ると分かった。ほかの明るい点（建物など）と形で見分けられるか。
方法:
    データ  tools/build_cr_chips.py の切り出し（48 × 48 の複素数、いちばん明るい画素の振幅でそろえる）。3 種類:
            2 = 反射板、1 = ほかの明るい点、0 = 背景。学習ではでたらめに 32 × 32 をずらして切り出し、全体の位相をでたらめに回す
    分け方  --split cr: 反射板ごとに分ける（テストの反射板は訓練で一度も見ない。同じ範囲のほかの点・背景も同じ側）
            --split date: 日付で分ける（新しい日付をテストに）
    条件（--model）:
      sin      Sin の板の光路（vendor の SN2-Parts-2D 0.11.0 の Net2D）。複素数の切り出しをそのまま光の場として入れる
      sin_amp  同じ光路に、振幅だけ（位相を捨てる）を入れる（位相が効いているかを見る）
      cnn      小さな CNN（実部・虚部の 2 チャネル）。学習する数を sin にそろえる
      cnn_amp  小さな CNN（振幅だけ）
      feat     手で作った特徴（明るい順の上位 10 画素の明るさ・平均）のロジスティック回帰（学習が軽い定番）
      scr      明るさの強さ（背景との比 dB）だけでしきい値（明るさをそろえる前の値。参考）
    水増し（--subbands K、持ち主の案）: 距離の向きの帯域を K 個に分けた「少し違う色のフレーム」を訓練に混ぜる（学習する条件だけ）
指標: テストの正解率、反射板の再現率と適合率、取り違えの表（真 × 予測）、学習する数、時間
結果: results/exp02.txt（1 回 = 1 行 JSON）。図は tools/fig_exp02.py
実行: CUDA_VISIBLE_DEVICES= .venv/Scripts/python experiments/exp02_cr_chips.py --model sin --split cr --seed 0
"""
from __future__ import annotations

import argparse
import json
import math
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
TEST_CR = ["N02K", "N05K", "N09K", "N12K", "N16K"]   # テストに回す反射板（訓練では見ない）


def load(split, seed, data="chips"):
    """切り出しを読み、訓練とテストに分ける。data = "chips"（1 日付の SLC）か "ifg_chips"（2 日付の干渉データ、実験04）。"""
    D = torch.load(data_dir() / "processed" / "cr_chips" / f"{data}.pt", weights_only=False)
    X, y = D["X"].to(torch.complex64), D["y"]             # 干渉データは倍精度で保存されているので、単精度にそろえる
    if split == "cr":
        te = torch.tensor([c in TEST_CR for c in D["cr"]])
    else:
        dates = sorted(set(D["date"]))
        last = dates[-max(1, len(dates) // 4):]            # 新しい 1/4 の日付をテストに
        te = torch.tensor([d in last for d in D["date"]])
    return X[~te], y[~te], X[te], y[te], D["scr_db"][~te], D["scr_db"][te]


def subband(X, K, rng):
    """持ち主の案（2026-10-31）の水増し: 距離の向きの帯域を K 個に分け、でたらめに選んだ 1 つ（または全部）だけを残す。

    SAR の像は、少しずつ違う電波の色（周波数）で見た像を重ねたもの。帯域の一部だけを残すと、
    「少し違う色で、少しぼけて見た同じ物」の像になる（物理的に正しい別のフレーム）。格子の大きさは変えない（周波数の側で外を 0 にする）。
    k = 0 は全部の帯域（元のまま）。
    """
    n, _, W = X.shape
    F_ = torch.fft.fftshift(torch.fft.fft(X, dim=-1), dim=-1)
    edges = torch.linspace(0, W, K + 1).round().long()
    k = torch.randint(0, K + 1, (n,), generator=rng)
    mask = torch.ones(n, 1, W)
    for i in range(n):
        if k[i] > 0:
            mask[i] = 0
            mask[i, :, edges[k[i] - 1]:edges[k[i]]] = 1
    out = torch.fft.ifft(torch.fft.ifftshift(F_ * mask, dim=-1), dim=-1)
    return out / (out.abs().amax((-2, -1), keepdim=True) + 1e-12)


def recenter(X):
    """切り出しごとに、周波数の中心（スペクトルの重心）を 0 に戻す = 位相の傾きを取り除く（TOPS の deramp の簡単な代わり）。

    Sentinel-1 の IW の SLC は、進行方向の周波数の中心が場所によって動く（位相が場所に比例して回る）。
    光の場で言えば「斜めから照らしている」のと同じで、レンズの後の像が横にずれる。Sin の板の窓の読み方は
    これに弱いはず（実験02 の考察）。重心は円の上の平均（e^{i2πk/N} の平均の向き）で求める。
    """
    n, H, W = X.shape
    P = torch.fft.fft2(X).abs() ** 2
    ky = torch.exp(2j * math.pi * torch.arange(H) / H)
    kx = torch.exp(2j * math.pi * torch.arange(W) / W)
    cy = torch.angle((P.sum(2) * ky).sum(1)) / (2 * math.pi) * H        # 進行方向の周波数の重心（周波数の番号）
    cx = torch.angle((P.sum(1) * kx).sum(1)) / (2 * math.pi) * W        # 距離の向き
    yy = torch.arange(H)[None, :, None].float()
    xx = torch.arange(W)[None, None, :].float()
    ramp = torch.polar(torch.ones(n, H, W), -2 * math.pi * (cy[:, None, None] * yy / H + cx[:, None, None] * xx / W))
    return X * ramp


def lowpass(X, n):
    """分解能を 1/n に落とす（周波数の側で、真ん中の 1/n の帯域だけ残す。格子の大きさは変えない）。10 m 相当の粗い画像の代わり。"""
    if n <= 1:
        return X
    F_ = torch.fft.fftshift(torch.fft.fft2(X), dim=(-2, -1))
    H, W = X.shape[-2:]
    m = torch.zeros(H, W)
    hh, hw = max(1, H // (2 * n)), max(1, W // (2 * n))
    m[H // 2 - hh:H // 2 + hh + 1, W // 2 - hw:W // 2 + hw + 1] = 1
    out = torch.fft.ifft2(torch.fft.ifftshift(F_ * m, dim=(-2, -1)))
    return out / (out.abs().amax((-2, -1), keepdim=True) + 1e-12)


def mask_center(z, r):
    """切り出しの真ん中の半径 r 画素（距離の向き。進行方向は画素が約 6 倍長いので r/6、最低 1 行）を 0 にする（周りだけを見せる）。"""
    if r <= 0:
        return z
    H, W = z.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(H) - H // 2, torch.arange(W) - W // 2, indexing="ij")
    ry = max(1.0, r / 6)
    m = ((yy / ry) ** 2 + (xx / r) ** 2) > 1
    return z * m


def crop(X, rng=None):
    """48 × 48 → 32 × 32。rng があればでたらめにずらし（±8 画素）、全体の位相を回す。なければ真ん中。"""
    if rng is None:
        o = (X.shape[-1] - L) // 2
        return X[:, o:o + L, o:o + L]
    n = len(X)
    oy = torch.randint(0, X.shape[-2] - L + 1, (n,), generator=rng)
    ox = torch.randint(0, X.shape[-1] - L + 1, (n,), generator=rng)
    ar = torch.arange(L)
    ri = (oy[:, None] + ar[None, :])[:, :, None]            # (n, L, 1) 行の番号
    ci = (ox[:, None] + ar[None, :])[:, None, :]            # (n, 1, L) 列の番号
    out = X[torch.arange(n)[:, None, None], ri, ci]           # 1 回の添え字で切り出す（1 個ずつの繰り返しより速い）
    ph = torch.rand(n, 1, 1, generator=rng) * 2 * math.pi
    return out * torch.polar(torch.ones_like(ph), ph)


class SmallCNN(nn.Module):
    """小さな CNN（比較相手）。入力は (N, ch, 32, 32)。"""

    def __init__(self, ch, width, n_classes=3):
        super().__init__()
        self.c1 = nn.Conv2d(ch, width, 5, padding=2)
        self.c2 = nn.Conv2d(width, width, 3, padding=1)
        self.fc = nn.Linear(width, n_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.relu(self.c2(x))
        return self.fc(x.amax((-2, -1)))


def prep(z, a):
    """隠す・粗くするの前処理（実験03）。

    隠したあと、残った周りの最大の振幅で割り直す。割り直さないと「隠した真ん中がとても明るかった」ことが
    周りの明るさの低さとして漏れてしまう（形ではなく、そろえ方のくせで当たってしまう）。
    """
    if a.mask <= 0:
        return z
    m = mask_center(z, a.mask)
    return m / (m.abs().amax((-2, -1), keepdim=True) + 1e-12)


def to_input(model, z):
    """複素数の切り出し (N, 32, 32) → 各モデルの入力。"""
    if model == "sin":
        return z
    if model == "sin_amp":
        return z.abs().to(torch.complex64)
    if model == "cnn":
        return torch.stack([z.real, z.imag], 1)
    return z.abs()[:, None]                                 # cnn_amp


def features(z):
    """手で作った特徴: 明るさ（|z|²）の上位 10 画素と平均（点らしさ = 光が 1 点に集まっているか）。"""
    p = (z.abs() ** 2).flatten(1)
    top = p.sort(1, descending=True).values[:, :10]
    return torch.cat([top, p.mean(1, keepdim=True)], 1)


def report(pred, yt):
    """正解率・反射板の再現率と適合率・取り違えの表。"""
    cm = torch.zeros(3, 3, dtype=torch.long)
    for t, p in zip(yt.tolist(), pred.tolist()):
        cm[t, p] += 1
    acc = (pred == yt).float().mean().item()
    rec = cm[2, 2].item() / max(1, cm[2].sum().item())
    prec = cm[2, 2].item() / max(1, cm[:, 2].sum().item())
    return dict(acc=round(100 * acc, 2), cr_recall=round(100 * rec, 2), cr_precision=round(100 * prec, 2), confusion=cm.tolist())


def main():
    ap = argparse.ArgumentParser(description="実験02: 角の反射板を見分ける")
    ap.add_argument("--model", required=True, choices=["sin", "sin_amp", "cnn", "cnn_amp", "feat", "scr"])
    ap.add_argument("--split", default="cr", choices=["cr", "date"])
    ap.add_argument("--data", default="chips", choices=["chips", "ifg_chips"],
                    help="ifg_chips = 2 日付の干渉データ（A·conj(B)）の切り出し（実験04、持ち主の方針: データどうしの相関だけ）")
    ap.add_argument("--sqrt-amp", action="store_true",
                    help="干渉データの振幅を平方根にする（|A||B| → √(|A||B|)。明るさの幅を 1 日付と同じにそろえ、位相はそのまま）")
    ap.add_argument("--plates", type=int, default=8)
    ap.add_argument("--patches", type=int, default=8)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--mask", type=float, default=0, help="実験03（持ち主の問い 2）: 真ん中の半径（距離の向きの画素）を隠し、周りだけで当てる。ずらしはしない")
    ap.add_argument("--recenter", action="store_true", help="切り出しごとに位相の傾きを取り除く（実験02 の考察の確かめ）")
    ap.add_argument("--lowpass", type=int, default=1, help="分解能を 1/n に落とす（10 m 相当の粗い画像の代わり）")
    ap.add_argument("--subbands", type=int, default=0, help="0 なら水増しなし。K なら距離の向きの帯域を K 個に分けて水増し（持ち主の案）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "exp02.txt"))
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    torch.set_num_threads(4)
    Xtr, ytr, Xte, yte, str_, ste = load(a.split, a.seed, a.data)
    if a.sqrt_amp:                                          # 掛け算で明るさの幅が 2 乗になる（尾が −21 → −42 dB）のを元に戻す
        Xtr = torch.polar(Xtr.abs().sqrt(), Xtr.angle())
        Xte = torch.polar(Xte.abs().sqrt(), Xte.angle())
    if a.recenter:
        Xtr, Xte = recenter(Xtr), recenter(Xte)
    Xtr, Xte = lowpass(Xtr, a.lowpass), lowpass(Xte, a.lowpass)
    jitter = a.mask <= 0                                    # 隠すときは、反射板が真ん中にあるようにずらさない
    rng = torch.Generator().manual_seed(a.seed)
    t0 = time.time()
    params = 0
    if a.model == "scr":                                    # しきい値 2 本を訓練で決める（背景 < t1 ≤ 明るい点 < t2 ≤ 反射板）
        best = (-1, 0, 0)
        for t1 in np.arange(0, 30, 1.0):
            for t2 in np.arange(t1, 50, 1.0):
                p = (str_ >= t1).long() + (str_ >= t2).long()
                best = max(best, ((p == ytr).float().mean().item(), t1, t2))
        pred = (ste >= best[1]).long() + (ste >= best[2]).long()
        params = 2
    elif a.model == "feat":
        f_tr, f_te = features(prep(crop(Xtr), a)), features(prep(crop(Xte), a))
        mu, sd = f_tr.mean(0), f_tr.std(0) + 1e-6
        lin = nn.Linear(f_tr.shape[1], 3)
        opt = torch.optim.Adam(lin.parameters(), lr=0.05)
        for _ in range(2000):
            loss = F.cross_entropy(lin((f_tr - mu) / sd), ytr)
            opt.zero_grad()
            loss.backward()
            opt.step()
        pred = lin((f_te - mu) / sd).argmax(1)
        params = sum(p.numel() for p in lin.parameters())
    else:
        if a.model.startswith("sin"):
            net = p2.Net2D(L, 3, plates=("stripes",) * a.plates, patches=a.patches, z=32.0, seed=a.seed, window=3)
        else:
            net = SmallCNN(2 if a.model == "cnn" else 1, 12)
        net = net.to(a.device)
        params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=0.01)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
        w = torch.bincount(ytr, minlength=3).float()
        w = (w.sum() / (3 * w)).to(a.device)                 # 種類の数の違いをならす重み
        for step in range(a.steps):
            idx = torch.randint(len(Xtr), (256,), generator=rng)
            xb = subband(Xtr[idx], a.subbands, rng) if a.subbands > 1 else Xtr[idx]
            z = prep(crop(xb, rng) if jitter else crop(xb) * torch.polar(torch.ones(len(xb), 1, 1), torch.rand(len(xb), 1, 1, generator=rng) * 6.2832), a)
            loss = F.cross_entropy(net(to_input(a.model, z).to(a.device)), ytr[idx].to(a.device), weight=w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
        with torch.no_grad():
            pred = torch.cat([net(to_input(a.model, prep(crop(Xte[i:i + 512]), a)).to(a.device)).argmax(1).cpu()
                              for i in range(0, len(Xte), 512)])
    rec = dict(vars(a), exp="04" if a.data == "ifg_chips" else ("03" if (a.mask > 0 or a.lowpass > 1) else "02"), params=int(params), train_s=round(time.time() - t0, 1),
               n_train=len(ytr), n_test=len(yte), test_cr=TEST_CR if a.split == "cr" else None,
               version="vendor sn2_parts_2d 0.11.0", torch=torch.__version__, region="oklahoma", **report(pred, yte))
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(json.dumps({k: rec[k] for k in ["model", "split", "seed", "params", "train_s", "acc", "cr_recall", "cr_precision", "confusion"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
