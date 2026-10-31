# vendor/ — 別のリポジトリから写したコード

| ファイル | 元のファイル | 元のコミット | ライセンス |
|---|---|---|---|
| sinwave2d_minimal_v0_3_0.py | [Project_Metasequoia](https://github.com/devaf09/Project_Metasequoia) の `sinwave2d_minimal.py`（v0.3.0、画像を見分ける SinWave2D） | 811494d | MIT（Copyright (c) 2026 devaf09） |
| sn2_parts_2d_v0_11_0.py | [SN2-Parts-2D](https://github.com/devaf09/SN2-Parts-2D) の `sn2_parts_2d.py`（v0.11.0、2 次元の部品） | 5238c39 | MIT（Copyright (c) 2026 devaf09） |

- 中身は書き換えていない。元のコミットのファイルとバイト単位で同じ（git の blob は 5dcdabf と 83a6ad9。2026-10-31 に `git hash-object` で確かめた）。
- 使うときは版番号つきの名前で import する（例: `import vendor.sn2_parts_2d_v0_11_0 as p2`）。
- 新しい版を使いたいときは、ここを直接直さず、元から写し直して新しい版番号のファイルを置き、この表を書き換える。古い版は、それを使った実験が再現できるよう残す。
- 元のライセンス全文は、このリポジトリの [LICENSE](../LICENSE)（同じ MIT・同じ著作権者）を参照。
