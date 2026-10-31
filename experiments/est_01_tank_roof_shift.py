"""est_01: 浮き屋根のずれの見積もり — 屋根が上下すると、SLC で角の反射点が何画素ずれ、雑音の中でどこまで測れるか

着想: 持ち主の案（2026-10-31）「位相差・畳み込みから 10 m より細かく」。docs/design_subpixel.md 2 節の概算を数で確かめる。
方法（作り物、GPU もダウンロードもなし）:
    幾何   屋根と壁の角（2 回はね返る点）は、屋根が Δh 上がると、斜めの距離が Δh·cosθ 縮む（θ = 入射角）
    装置   Sentinel-1 IW の SLC: 距離の画素 2.33 m、距離の分解能 ≈ 2.7 m（帯域で決まる。知: 公式の値は M1 で確かめる）
           点の写り方 = sinc（帯域が四角の場合。窓関数ありは少し太る）
    雑音   周りの地面の斑点（スペックル）= 複素数の正規分布。点の強さとの比 SCR（dB）を変える
    推定   1 次元の距離の向きで、8 倍に補間して明るさの山を探し、頂上を放物線で細かく決める（素朴なやり方）
指標: 位置の誤差（標準偏差、m）→ 屋根の高さの誤差（m）= 位置の誤差 / cosθ。タンクの高さ（約 15 m）と比べる
結果: results/est01.txt（1 行 JSON）。実行: CUDA_VISIBLE_DEVICES= .venv/Scripts/python experiments/est_01_tank_roof_shift.py --seed 0
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")


def simulate(scr_db, n_trials, rng, pix=2.33, res=2.7, n=64, up=8):
    """点（位置は試行ごとに画素の中のでたらめな位置）+ 斑点の雑音を作り、山の位置を推定して誤差を返す（m）。"""
    x = (np.arange(n) - n / 2) * pix                       # 画素の中心の距離（m）
    amp = 10 ** (scr_db / 20)                              # 点の振幅（雑音の 1 画素あたりの平均の強さを 1 とする）
    errs = np.empty(n_trials)
    for t in range(n_trials):
        shift_m = rng.uniform(-pix / 2, pix / 2)          # 画素の格子に対する点の位置（試行ごとに変える）
        sig = amp * np.sinc((x - shift_m) / res) * np.exp(1j * rng.uniform(0, 2 * np.pi))
        clutter = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / math.sqrt(2)
        # 斑点も装置の帯域を通るので、同じぼけで畳み込む（周波数の側で帯域の外を消す）
        cf = np.fft.fft(clutter)
        f = np.fft.fftfreq(n, d=pix)
        cf[np.abs(f) > 0.5 / res] = 0
        clutter = np.fft.ifft(cf) * math.sqrt(n / max(1, (np.abs(f) <= 0.5 / res).sum()))
        s = sig + clutter
        # 8 倍に補間（周波数の側でゼロを詰める = 帯域の内側だけの正しい補間）
        S = np.fft.fftshift(np.fft.fft(s))
        pad = np.zeros(n * up, dtype=complex)
        pad[(n * up - n) // 2:(n * up + n) // 2] = S
        su = np.fft.ifft(np.fft.ifftshift(pad))
        xu = (np.arange(n * up) - n * up / 2) * pix / up
        win = np.abs(xu) < 3 * res                         # 探すのは真ん中付近だけ（角の点の大まかな位置は分かっている前提）
        m = np.abs(su) * win
        k = int(np.argmax(m))
        # 山の頂上を、隣の 2 点と合わせた放物線で細かく決める（補間の格子 pix/up の刻みの誤差を消す）
        y0, y1, y2 = m[k - 1], m[k], m[k + 1]
        d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2) if (y0 - 2 * y1 + y2) != 0 else 0.0
        errs[t] = xu[k] + d * pix / up - shift_m
    return errs


def main():
    ap = argparse.ArgumentParser(description="est_01: 浮き屋根のずれの見積もり")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "est01.txt"))
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--theta-deg", type=float, default=39.0, help="入射角（クッシングの上昇軌道 34 の辺りの見込み）")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    cos_t = math.cos(math.radians(a.theta_deg))
    print(f"幾何: 屋根 10 m の上下 → 斜めの距離 {10 * cos_t:.2f} m のずれ = SLC の距離の画素 {10 * cos_t / 2.33:.2f} 個分")
    with open(a.out, "a", encoding="utf-8") as fo:
        for scr in [0, 5, 10, 15, 20, 25, 30]:
            e = simulate(scr, a.trials, rng)
            rmse = float(np.sqrt(np.mean(e ** 2)))
            big = float(np.mean(np.abs(e) > 2.7))               # 分解能より大きく外した割合（別の山を拾った）
            rec = dict(exp="est01", seed=a.seed, trials=a.trials, theta_deg=a.theta_deg, scr_db=scr,
                       pos_rmse_m=round(rmse, 3), height_rmse_m=round(rmse / cos_t, 3), outlier_rate=round(big, 4))
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"SCR {scr:>2} dB: 位置の誤差 {rmse:6.2f} m → 屋根の高さの誤差 {rmse / cos_t:6.2f} m（大外れ {big:.1%}）")


if __name__ == "__main__":
    main()
