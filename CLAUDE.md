# YaneuraOu フォーク運用メモ

## NNUE 新アーキテクチャの設計仕様

学習器 tatara (フォーク `yjbg9201/tatara`) と同時に進める NNUE 新アーキテクチャの
実装は [docs/nnue-architecture-ideas.md](docs/nnue-architecture-ideas.md) に従う。
同じ文書が tatara 側の `docs/` にもあり、tatara 側の判断記録とあわせて読む。

- 1 セッションで扱うのは仕様書のアイデア 1 つ (ID で指定) だけ。
- 仕様書と実ソースが食い違ったら実ソースを優先し、差異を報告する。
- 仕様書の ID (`A2` 等) はコードの識別子・ファイル名に使わない。特に
  `source/eval/nnue/features/a2.h` には upstream 由来の別物の特徴量 `Features::A2`
  (玉を含む全駒の BonaPiece) が既にあるので、名前が衝突しないようにする。
- 既存アーキテクチャ (`source/eval/nnue/architectures/*.h`) の挙動は変えず、新アーキは
  ビルドフラグで切り替える。

## ビルド

`source/Makefile` を使う (`YANEURAOU_EDITION` と `TARGET_CPU` を指定する)。
