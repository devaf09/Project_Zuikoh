"""比較相手の検出器（zuikoh.detect）の確認。"""
import numpy as np

from zuikoh.detect import ca_cfar, disk_template, peaks, template_match


def test_cfar_false_alarm_rate():
    """背景だけ（指数分布の斑点）のとき、誤って物とする割合が pfa に近い（桁が合う）。"""
    rng = np.random.default_rng(0)
    img = rng.exponential(1.0, (400, 400))
    rate = ca_cfar(img, guard=2, train=6, pfa=1e-3).mean()
    assert 3e-4 < rate < 3e-3


def test_cfar_finds_bright_points():
    """背景より 20 dB 明るい点 5 個を全部見つけ、位置が合う。"""
    rng = np.random.default_rng(1)
    img = rng.exponential(1.0, (200, 200))
    pts = [(30, 40), (60, 150), (100, 100), (150, 30), (180, 180)]
    for r, c in pts:
        img[r, c] = 100.0
    found = peaks(ca_cfar(img, pfa=1e-6) * img, min_dist=3)
    assert {tuple(p) for p in found} >= set(pts)


def test_template_match_peak_at_disk():
    """円の見本との相関は、円を置いた場所で最大になる。"""
    img = np.zeros((80, 80))
    y, x = np.mgrid[:80, :80]
    img[np.hypot(y - 50, x - 30) <= 4] = 1.0
    s = template_match(img, disk_template(8))
    r, c = np.unravel_index(np.argmax(s), s.shape)
    assert abs(r - 50) <= 1 and abs(c - 30) <= 1
