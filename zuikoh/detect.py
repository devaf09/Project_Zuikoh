"""zuikoh.detect — 比較相手の検出器（M1）: CFAR と、テンプレートの照合。

    from zuikoh.detect import ca_cfar, peaks, template_match
    mask = ca_cfar(img, guard=3, train=8, pfa=1e-4)          # 誤報の割合を一定に保つしきい値を超えた画素
    pts = peaks(mask * img, min_dist=3)                      # 塊ごとにいちばん明るい画素の位置（行, 列）の列
    score = template_match(img, tmpl)                        # 見本の形との相関（4f 相関器と同じ計算）

言葉:
- CFAR（Constant False Alarm Rate）= 調べる画素の周りの「背景の輪」の平均から、誤報の割合 pfa が一定になる
  しきい値を決め、それを超えた画素を物とする。SAR の定番の検出。ここでは「セル平均」の形（CA-CFAR）
- 背景の明るさは、斑点のため指数分布（1 回分の画像）に従うと仮定する。平均した画像（RTC）では少し保守的になる
- テンプレートの照合 = 見本の形をずらしながら重なり具合を測る。光ならレンズ → 周波数の面の板 → レンズ（4f 系）で一度にできる

ライセンス: MIT
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def ca_cfar(img: np.ndarray, guard: int = 3, train: int = 8, pfa: float = 1e-4) -> np.ndarray:
    """セル平均の CFAR。img は明るさ（線形、dB ではない）。物と判定した画素を True で返す。

    guard: 調べる画素の周りで、物の一部かもしれないので背景に入れない幅（画素）
    train: その外側で背景として平均する輪の幅（画素）
    pfa  : 背景だけのときに誤って物とする割合
    """
    img = np.nan_to_num(img.astype("float64"), nan=0.0)
    big, small = 2 * (guard + train) + 1, 2 * guard + 1
    s_big = ndimage.uniform_filter(img, big, mode="reflect") * big ** 2      # 外の四角の合計
    s_small = ndimage.uniform_filter(img, small, mode="reflect") * small ** 2  # 守りの四角の合計
    n = big ** 2 - small ** 2
    mean_bg = (s_big - s_small) / n                                            # 背景の輪の平均
    alpha = n * (pfa ** (-1.0 / n) - 1.0)                                      # 指数分布のときの CA-CFAR の係数
    return img > alpha * mean_bg


def peaks(score: np.ndarray, min_dist: int = 3) -> np.ndarray:
    """0 より大きい塊ごとに、いちばん明るい画素の位置を返す（(個数, 2) の 行・列）。近すぎる山は 1 つにまとめる。"""
    lab, n = ndimage.label(score > 0)
    if n == 0:
        return np.zeros((0, 2), dtype=int)
    pos = np.array(ndimage.maximum_position(score, lab, range(1, n + 1)), dtype=int)
    val = score[pos[:, 0], pos[:, 1]]
    keep = []
    for i in np.argsort(-val):                                                 # 明るい順に、近くの山を消しながら残す
        if all(np.hypot(*(pos[i] - pos[j])) >= min_dist for j in keep):
            keep.append(i)
    return pos[keep]


def disk_template(diam_px: float, size: int | None = None) -> np.ndarray:
    """直径 diam_px の明るい円の見本（平均 0・大きさ 1 にそろえる）。タンクの大まかな形。"""
    size = size or int(np.ceil(diam_px)) + 4
    y, x = np.mgrid[:size, :size] - (size - 1) / 2
    t = (np.hypot(y, x) <= diam_px / 2).astype("float64")
    t -= t.mean()
    return t / np.linalg.norm(t)


def template_match(img: np.ndarray, tmpl: np.ndarray) -> np.ndarray:
    """見本との相関（FFT で計算 = 4f 系の光学相関器と同じ）。img は dB などの見やすい尺度で渡す。"""
    img = np.nan_to_num(img.astype("float64"), nan=float(np.nanmedian(img)))
    img = img - ndimage.uniform_filter(img, tmpl.shape[0] * 3)                 # ゆっくりした明るさの変化を除く
    H, W = img.shape
    th, tw = tmpl.shape
    F = np.fft.rfft2(img, s=(H + th, W + tw))
    G = np.fft.rfft2(tmpl[::-1, ::-1], s=(H + th, W + tw))
    full = np.fft.irfft2(F * G, s=(H + th, W + tw))
    return full[th // 2:th // 2 + H, tw // 2:tw // 2 + W]
