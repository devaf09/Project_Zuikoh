"""sinwave2d_minimal.py — 正弦の干渉と回折だけで作る画像分類器（単独で動く 1 ファイル版、2 次元）

このファイルだけをコピーすれば使えます（依存は PyTorch のみ）。実験用のオプションは入れず、
段階 A の実験（NOTES.md、実験 A0〜A8）で分かっている「最も効率の良い構成」だけを実装しています。
1 次元版 sinwave_minimal.py（文字列の分類。このリポジトリでは vendor/sinwave_minimal_v1_3_0.py）の 2 次元版です。

    import sinwave2d_minimal as sw

    net = sw.SinWave2DNet(own=[0, 1, 2, 3, 4])            # 窓 10 個のうち 0〜4 番を担当する技能
    sw.train(net, images, labels)                          # images: (n, 28, 28) の 0〜1、labels は窓の番号
    sw.predict(net, images)                                # → 窓の番号

    merged = sw.merge([net_a, net_b])                      # 別々に学習した技能を足すだけで融合
    merged = sw.merge([net_a])                             # 外せば完全に元どおり（取消）

しくみ（光の実験装置に例えると）
    画像（明るい画素ほど光を通すマスク）→ 滲み層 → [Sin の板 → 回折] × 16 → レンズ → スクリーンの窓 → 答え
    学習するのは「Sin の板」の模様（区画ごとの 振幅 A と 初期位相 φ）と、窓ごとの背景光・全体の温度だけ。

1 次元版との違い（実験で分かったこと）
    - 照明（tilt）は使わない: 2 次元では干渉を減らしても精度が上がらず、計算だけ増える（実験 A3）
    - 画像は拡大しない: 拡大は伝搬距離を変えるのと同じ働きで、計算が増えるだけ（実験 A2・A8）
    - 典型の模様での判定は窓より 10 ポイント以上悪いので入れない（実験 A1・A6）

変更履歴（成果が出たらここを更新する）
    0.3.0 (2026-10-26)  merge(nets, weights=...): 技能ごとの重み付きの平均 Σ w·s / Σ w（重み 0 は外したのと完全に同じ）。
                        save(..., weaver=...): 提供者の名前を見出しに入れる（load した技能の net.weaver で読める）。
                        どちらも SinCrest（信頼のマップ）から段階 C の融合への受け渡しのため（掲示板 2026-10-26 の SinCrest のお願い）。
                        学習器の中身と性能は 0.2.0 と同じ。
    0.2.0 (2026-10-22)  保存形式を safetensors 形式に（pickle を使わない。PyTorch だけで読み書きし、safetensors のライブラリでも読める）。
                        光学設定の指紋 fingerprint（設定と固定の縞 ω から作る）を保存し、読み込みと融合で一致を確かめる。
                        read_header で数値を読まずに版・設定・指紋を確かめられる。学習器の中身と性能は 0.1.0 と同じ。
    0.1.0 (2026-10-15)  最初の版。場 32×32、振幅マスク、滲み層 1 画素、区画ごとの Sin の板（ω 固定）、
                        2 次元の角スペクトル法、レンズ後の円周上の窓 10 個、背景光、並列光路の融合。
                        性能（実験用の全オプション版 sinwave2d/ での同じ構成、10000 steps、3 seed）:
                          区画 8×8 (2,059 個): MNIST 96.7±0.1%、Fashion-MNIST 86.7±0.2%
                          区画 16×16 (8,203 個、滲みなし): MNIST 96.7±0.1%、Fashion-MNIST 87.3±0.1%
                        1 step 約 0.02 秒、ピークメモリ 92MB（RTX 3060、バッチ 256）。
                        注意: 融合は 2 次元ではまだ弱い。0〜4 と 5〜9 を別々に学習して融合すると、10 クラスで
                        MNIST 81.4±2.2%、Fashion 53.1±1.1%（一度に学習なら 95.6 / 85.7。実験 A7、区画 8×8・滲みなし）。

ライセンス: MIT
"""
from __future__ import annotations

import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__version__ = "0.3.0"


class SinWave2DNet(nn.Module):
    """1 本の光路（= 1 つの技能）。

    n_slots : スクリーンに並べる窓の総数。融合する技能どうしで揃える
    own     : この技能が担当する窓の番号（クラス）。融合する技能どうしで重ならないようにする
    N       : 場の大きさ（N×N 画素）。画像 28×28 を中央に置く
    layers  : Sin の板の枚数
    patches : 板を patches×patches の区画に分け、区画ごとに (振幅, 初期位相) を学習する
    smear   : 滲み層のずらし幅（画素）。0 で無し
    seed    : 板の初期模様の乱数
    """

    def __init__(self, n_slots: int = 10, own: list[int] | None = None, N: int = 32, layers: int = 16,
                 patches: int = 8, smear: int = 1, z: float = 32.0, seed: int = 0):
        super().__init__()
        if N < 28:
            raise ValueError("N は 28 以上（画像 28×28 が場に収まること）")
        self.n_slots, self.N, self.smear = n_slots, N, smear
        self.own = list(own) if own is not None else list(range(n_slots))
        if not self.own or max(self.own) >= n_slots or len(set(self.own)) != len(self.own):
            raise ValueError("own は n_slots 未満の、重複しない番号の列")
        self.config = dict(n_slots=n_slots, own=self.own, N=N, layers=layers, patches=patches, smear=smear,
                           z=z, seed=seed)
        self.off = (N - 28) // 2

        # --- Sin の板（学習する）: 区画 p は位置 (x, y) の光の位相を A·sin(ωx·x + ωy·y + φ) だけ遅らせる ---
        # ω（縞の向きと細かさ）は乱数で決めて固定する（学習しても同等: 実験 A4）
        g = torch.Generator().manual_seed(seed)
        r = torch.rand(4, layers, patches * patches, generator=g)
        lo, hi = math.log(2 * math.pi / N), math.log(math.pi)
        mag, ang = torch.exp(lo + (hi - lo) * r[1]), 2 * math.pi * r[2]
        self.A = nn.Parameter(0.5 + 0.5 * r[0])                       # (layers, patches²) 振幅
        self.phi = nn.Parameter(2 * math.pi * r[3])                   # (layers, patches²) 初期位相
        self.register_buffer("wx", mag * torch.cos(ang))
        self.register_buffer("wy", mag * torch.sin(ang))
        idx = torch.arange(N) * patches // N
        self.register_buffer("pidx", (idx[:, None] * patches + idx[None, :]).reshape(-1))   # 画素 → 区画
        yy, xx = torch.meshgrid(torch.arange(N, dtype=torch.float32), torch.arange(N, dtype=torch.float32),
                                indexing="ij")
        self.register_buffer("xx", xx.reshape(-1))
        self.register_buffer("yy", yy.reshape(-1))

        # --- 回折（固定）: 2 次元の角スペクトル法。波長 1 画素、板の間隔 z 画素、周期境界 ---------------
        k = 2 * math.pi * torch.fft.fftfreq(N)
        kz = torch.sqrt((2 * math.pi) ** 2 - k[:, None] ** 2 - k[None, :] ** 2)
        self.register_buffer("H", torch.polar(torch.ones(N, N), kz * z))

        # --- 窓（固定）: レンズ後の面で、中心から半径 N/4 の円周上に 3×3 画素の窓を n_slots 個 ---------
        det = torch.zeros(n_slots, N, N)
        for c in range(n_slots):
            t = 2 * math.pi * c / n_slots
            cy, cx = round(N / 2 + N / 4 * math.sin(t)), round(N / 2 + N / 4 * math.cos(t))
            det[c, cy - 1:cy + 2, cx - 1:cx + 2] = 1 / 9
        self.register_buffer("det", det.reshape(n_slots, -1))

        # --- 窓の読み方（学習する）: 温度と、窓ごとの背景光 ----------------------------------------
        self.gain = nn.Parameter(torch.tensor(5.0))
        self.bias = nn.Parameter(torch.zeros(n_slots))

    def intensity(self, x: torch.Tensor) -> torch.Tensor:
        """画像 (n, 28, 28) → スクリーンの明るさ (n, N·N)、平均 1。"""
        N, n = self.N, x.shape[0]
        m = F.pad(x, (self.off, N - self.off - 28, self.off, N - self.off - 28))
        u = torch.complex(m, torch.zeros_like(m))                     # 明るい画素ほど光を通す
        if self.smear:                                                # 滲み層: 上下左右にずらした場との平均
            s = self.smear
            u = (u + u.roll(s, 1) + u.roll(-s, 1) + u.roll(s, 2) + u.roll(-s, 2)) / 5
        u = u.reshape(n, -1)
        p = self.pidx
        for d in range(self.A.shape[0]):
            phase = self.A[d, p] * torch.sin(self.wx[d, p] * self.xx + self.wy[d, p] * self.yy + self.phi[d, p])
            u = u * torch.polar(torch.ones_like(phase), phase)       # Sin の板
            u = torch.fft.ifft2(torch.fft.fft2(u.reshape(n, N, N)) * self.H).reshape(n, -1)   # 回折
            u = u / (u.abs().pow(2).mean(-1, keepdim=True).sqrt() + 1e-12)                  # 増幅
        u = torch.fft.fftshift(torch.fft.fft2(u.reshape(n, N, N), norm="ortho"), dim=(-2, -1))  # レンズ
        inten = u.abs().pow(2).reshape(n, -1)
        return inten / (inten.mean(-1, keepdim=True) + 1e-12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """画像 (n, 28, 28) → 窓ごとのスコア (n, n_slots)。"""
        return self.gain * (self.intensity(x) @ self.det.T) + self.bias


def train(net: SinWave2DNet, images: torch.Tensor, labels: torch.Tensor, steps: int = 10000, lr: float = 0.01,
          batch_size: int = 256, seed: int = 0, verbose: bool = False) -> SinWave2DNet:
    """labels は窓の番号（net.own のどれか）。損失は窓すべてで取るので、担当外の窓は暗くなるように学ぶ。"""
    labels = torch.as_tensor(labels)
    if len(images) == 0 or len(images) != len(labels):
        raise ValueError("images と labels は同じ長さで、空でないこと")
    if set(labels.tolist()) - set(net.own):
        raise ValueError(f"labels は own {net.own} のどれかにする")
    dev = net.det.device
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    gen = torch.Generator().manual_seed(seed)
    net.train()
    for step in range(steps):
        i = torch.randint(len(labels), (min(batch_size, len(labels)),), generator=gen)
        loss = F.cross_entropy(net(images[i].to(dev)), labels[i].to(dev))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if verbose and step % max(steps // 10, 1) == 0:
            print(f"step {step:5d}  loss {loss.item():.4f}")
    return net


@torch.no_grad()
def scores(net, images: torch.Tensor, batch_size: int = 500) -> torch.Tensor:
    """各窓のスコア (n, n_slots)。担当外の窓は -inf。net は SinWave2DNet でも Merged でもよい。"""
    net.eval()
    dev = net.det.device
    out = torch.cat([net(images[i:i + batch_size].to(dev)) for i in range(0, len(images), batch_size)])
    mask = torch.full((net.n_slots,), float("-inf"), device=out.device)
    mask[net.own] = 0
    return (out + mask).cpu()


def predict(net, images: torch.Tensor) -> torch.Tensor:
    """一番明るい（スコアの高い）担当の窓の番号。"""
    return scores(net, images).argmax(1)


class Merged(nn.Module):
    """別々に学習した技能（光路）を並べたもの。互いに干渉しない光（色や偏光の違い）を想定し、
    スクリーンでは明るさを足すだけ（ここでは平均）。追加の学習は要らず、外せば完全に元に戻る。"""

    def __init__(self, nets: list[SinWave2DNet], weights=None):
        super().__init__()
        if not nets:
            raise ValueError("技能が空")
        owned = [o for n in nets for o in n.own]
        if len(set(owned)) != len(owned):
            raise ValueError("融合する技能どうしで担当する窓 (own) が重なっている")
        if len({(n.n_slots, n.N) for n in nets}) != 1:          # 並列の光路は別の装置なので、そろえるのはスクリーン（窓）だけ
            raise ValueError("融合する技能どうしで n_slots と N を揃える")
        w = [1.0] * len(nets) if weights is None else [float(v) for v in weights]
        if len(w) != len(nets) or min(w) < 0 or sum(w) <= 0:
            raise ValueError("weights は技能と同じ数の 0 以上の数で、和が正")
        self.nets = nn.ModuleList(nets)
        self.weights = w
        self.n_slots, self.own, self.det = nets[0].n_slots, owned, nets[0].det

    def forward(self, x):
        # 重み付きの平均 Σ w_i s_i / Σ w_i。重み 0 の技能は、外したのと完全に同じになる（取消の厳密さを保つ）
        tot = sum(self.weights)
        return sum(w * n(x) for w, n in zip(self.weights, self.nets) if w > 0) / tot


def merge(nets: list[SinWave2DNet], weights=None) -> Merged:
    """融合。取消は、外したい技能を除いたリストで merge し直せばよい（残りは元と完全に同じ）。
    weights を渡すと技能ごとの重み付きの平均になる（例: 提供者の信頼、SinCrest の layer_trust の重み）。
    注意: 重みだけでは、他人の担当の入力に反応する技能（取り違え）は止められない（SinCrest 実験09）。
    担当なしの学習（実験10・15）と組み合わせて使う。"""
    return Merged(list(nets), weights)


# 保存形式（段階 C の規格の準備）: safetensors 形式（https://github.com/huggingface/safetensors）を PyTorch だけで読み書きする。
# 中身は「8 バイトの見出しの長さ + JSON の見出し + 数値の並び」だけで、pickle（読み込むとプログラムが動きうる形式）を使わない。
# 見出しの __metadata__ に、版・設定・光学設定の指紋を文字列で入れる。safetensors のライブラリでもそのまま読める。

_DTYPES = {torch.float32: "F32", torch.float64: "F64", torch.int64: "I64"}
_TORCH = {v: k for k, v in _DTYPES.items()}
_OPTICS_KEYS = ("n_slots", "N", "layers", "patches", "smear", "z", "seed")     # 融合する技能どうしでそろえる設定


def fingerprint(net: SinWave2DNet) -> str:
    """光学設定の指紋（16 進 16 文字）。設定（担当の窓 own を除く）と、乱数で決めた固定の縞 ω の値から作る。
    読み込むときに、設定から作り直した装置が保存したときと同じかを確かめるのに使う（乱数の作り方や版が変わると違う値になる）。
    並列の光路の融合では装置は別々でよい（そろえるのはスクリーンの窓だけ）。"""
    import hashlib
    cfg = {k: net.config[k] for k in _OPTICS_KEYS}
    w = torch.stack([net.wx, net.wy]).detach().cpu().double()
    text = json.dumps(dict(model="SinWave2DNet", wavelength=1.0, **cfg), sort_keys=True) +         ",".join(f"{v:.6e}" for v in w.flatten().tolist())
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def save(net: SinWave2DNet, path: str, weaver: str | None = None) -> None:
    """学習した値（A, φ, 温度, 背景光）と設定・指紋だけを safetensors 形式で保存する（区画 8×8 で約 8KB）。
    ω などの固定の部品は保存せず、設定（seed を含む）から作り直す。
    weaver: 技能の提供者の名前（任意）。見出しの __metadata__ に入れる（分散学習で、誰のレイヤかを記録するため）。"""
    state = {k: v.detach().cpu().contiguous() for k, v in net.state_dict().items() if k in ("A", "phi", "gain", "bias")}
    header, chunks, off = {}, [], 0
    for k in sorted(state):
        t = state[k]
        b = bytes(t.clone().untyped_storage())                       # 数値の並び（リトルエンディアン）
        header[k] = dict(dtype=_DTYPES[t.dtype], shape=list(t.shape), data_offsets=[off, off + len(b)])
        chunks.append(b)
        off += len(b)
    header["__metadata__"] = dict(format="sinwave2d", version=__version__, config=json.dumps(net.config),
                                  fingerprint=fingerprint(net), **({"weaver": str(weaver)} if weaver else {}))
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)                                         # 見出しを 8 バイトの区切りにそろえる
    with open(path, "wb") as f:
        f.write(len(h).to_bytes(8, "little"))
        f.write(h)
        for b in chunks:
            f.write(b)


def read_header(path: str) -> dict:
    """保存ファイルの見出し（JSON）だけを読む。中身の数値を読まずに、版・設定・指紋を確かめられる。"""
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        return json.loads(f.read(n))


def load(path: str, device=None) -> SinWave2DNet:
    """save で保存した技能を読み込む。設定から作り直した装置の指紋が、保存した指紋と違えば止める。"""
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
        raw = f.read()
    meta = header.pop("__metadata__")
    if meta.get("format") != "sinwave2d":
        raise ValueError("sinwave2d の保存ファイルではない")
    net = SinWave2DNet(**json.loads(meta["config"]))
    if fingerprint(net) != meta["fingerprint"]:
        raise ValueError("光学設定の指紋が合わない（作り直した装置が保存したときと違う）")
    state = {}
    for k, v in header.items():
        a, b = v["data_offsets"]
        state[k] = torch.frombuffer(bytearray(raw[a:b]), dtype=_TORCH[v["dtype"]]).reshape(v["shape"])
    net.load_state_dict(state, strict=False)
    net.weaver = meta.get("weaver")                                   # 提供者の名前（保存したときに渡していれば）
    return net.to(device) if device is not None else net


# 動作例: python sinwave2d_minimal.py（data/ に MNIST があること。torchvision を使う）
if __name__ == "__main__":
    import torchvision as tv

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr, te = tv.datasets.MNIST("data", train=True), tv.datasets.MNIST("data", train=False)
    X, Y, Xt, Yt = tr.data.float() / 255, tr.targets, te.data.float() / 255, te.targets
    a = train(SinWave2DNet(own=[0, 1, 2, 3, 4]).to(dev), X[Y < 5], Y[Y < 5], steps=2000)
    b = train(SinWave2DNet(own=[5, 6, 7, 8, 9], seed=1).to(dev), X[Y >= 5], Y[Y >= 5], steps=2000, seed=1)
    acc = lambda net, sel: (predict(net, Xt[sel]) == Yt[sel]).float().mean().item()
    print(f"技能 A 単独 (0〜4): {acc(a, Yt < 5):.3f}   技能 B 単独 (5〜9): {acc(b, Yt >= 5):.3f}")
    print(f"融合 (10 クラス): {acc(merge([a, b]), Yt >= 0):.3f}")
    print(f"B を外す: {acc(merge([a]), Yt < 5):.3f}（A 単独と同じ）")
    print(f"学習するパラメータ: {sum(p.numel() for p in a.parameters())} 個")
