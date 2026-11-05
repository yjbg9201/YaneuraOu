// NNUE評価関数の差分計算用のクラス

#ifndef CLASSIC_NNUE_ACCUMULATOR_H_
#define CLASSIC_NNUE_ACCUMULATOR_H_

#include "../../config.h"

#if defined(EVAL_NNUE)

#include "nnue_architecture.h"

namespace YaneuraOu {
namespace Eval::NNUE {

// 入力特徴量をアフィン変換した結果を保持するクラス
// 最終的な出力である評価値も一緒に持たせておく
// AVX-512命令を使用する場合に64bytesのアライメントが要求される。
struct alignas(64) Accumulator {
  std::int16_t
      accumulation[2][kRefreshTriggers.size()][kTransformedFeatureDimensions];
  Value score = VALUE_ZERO;
  bool computed_accumulation = false;
  bool computed_score = false;
#if defined(NNUE_HAS_NYUGYOKU_SCALAR_INPUT)
  // 玉ごと (添字は玉の持ち主の色) の、入玉スカラーの L1 寄与の表引きに使う値。
  // 直前の手で変わり得ない玉は親局面の値を引き継ぐ (evaluate_nnue.cpp の TransformInput)。
  Features::NyugyokuScalar::L1Inputs nyugyoku_inputs[COLOR_NB];
  bool computed_nyugyoku = false;
#endif
};

} // namespace Eval::NNUE
} // namespace YaneuraOu

#endif  // defined(EVAL_NNUE)

#endif
