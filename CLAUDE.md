# YaneuraOu（フォーク）作業メモ

## NNUE 新アーキテクチャ作業

学習器 tatara（`yjbg9201/tatara`）と同時に改造している NNUE 新アーキテクチャの
仕様・アイデア一覧・共通規約は [docs/nnue-architecture-ideas.md](docs/nnue-architecture-ideas.md)
にある。NNUE の特徴量・アーキテクチャ・評価関数ファイル形式に触れる作業の前に読むこと。

- 1 回の作業では 1 アイデア（ID 指定）だけを扱う（仕様書冒頭の指示）。
- 仕様書と実ソースが食い違う場合は実ソースを優先し、差異を報告する。
- 個別アイデアの調査結果・決定事項は tatara 側の `docs/adr/` に置いている。
- 学習器側と index・ハッシュ・量子化を揃える必要がある（仕様書 §1.1）。
  `FeatureSet<A, B>` の index 空間は宣言の逆順（後ろの `B` が低位）になる。

## ビルドと確認に使えるコマンド

- Linux での NNUE（HalfKP 256x2-32-32）ビルド:
  `cd source && make -j4 normal TARGET_CPU=AVX2 YANEURAOU_EDITION=YANEURAOU_ENGINE_NNUE COMPILER=clang++`
  （実行ファイルは `source/YaneuraOu-by-gcc` に出て `.gitignore` 対象外なのでコミットしないこと。
  オブジェクトはリポジトリ直下の `obj/` に出る）
- `isready` 後に使える確認用コマンド:
  - `e`: 現局面の `evaluate()` の値（生の評価値）を表示する
  - `test nn test_features`: ランダム対局で差分 index と全計算 index の一致を検査する
  - `test nn test_accumulator [games]`: 差分更新・キャッシュ経由の refresh と全再計算の一致を検査する
- 評価関数ファイルのハッシュ不一致は警告だけでロードは続行される
  （`source/eval/nnue/evaluate_nnue.cpp`）。サイズ違いはファイル末尾の検査で失敗する。
