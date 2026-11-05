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

- 入玉スカラーつきの NNUE (HalfKP 256x2-32-32 の L1 入力に入玉スカラー 96 個を連結):
  `YANEURAOU_EDITION=YANEURAOU_ENGINE_NNUE_HALFKP_NYUGYOKU_256X2_32_32`。ネットに固定する
  入玉ルールと符号化は `NYUGYOKU_RULE=csa27|csa24` と `NYUGYOKU_ENCODING=thermometer|onehot`
  (学習器 tatara の `simple --nyugyoku-rule` / `--nyugyoku-encoding` と同じ値にする)。

## テスト

- NNUE ビルドの `unittest` と `test nn ...` は局面評価を伴うので、先に `isready` で評価関数
  ファイルを読み込む。ネットが無ければ `source/eval/nnue/architectures/nnue_dummy_gen.py` の
  ダミーか、tatara の `simple_quantised_eval random-net` で作った乱数ネットを使う。
- `test nn eval_sfens <in.sfen> <out.tsv>`: 局面ごとに出力層の生の値と評価値を書く
  (tatara の量子化 CPU forward との integer-exact 突合用)。
- `test nn gen_sfens <count> <seed> <out.sfen> [king_walk_percent]`: ランダムプレイアウトの局面
  (一部の対局は玉を前に進めて入玉局面を含める)。
- `test nn bench_scalar <in.sfen> [repeat] [full|states|write]` / `test nn bench_eval <in.sfen>`:
  入玉スカラー 1 回 (内訳: 玉の状態の計算・96 次元の書き込み)・差分更新つきの評価 1 回の時間。
- `test nn test_nyugyoku_reuse [games]`: 入玉スカラーの L1 入力を親局面から引き継ぐ経路が毎回
  計算した値と一致するかをランダムプレイアウト (玉を前に進める対局・null move を含む) で確かめる。
