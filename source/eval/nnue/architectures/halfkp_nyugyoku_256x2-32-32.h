// HalfKP 256x2-32-32 の L1 入力に入玉スカラー 96 個を連結したアーキテクチャ
// (L1 入力 = 256 × 2 + 96 = 608)。
//
// 入玉ルール・符号化はビルドフラグで選び (NNUE_NYUGYOKU_RULE_CSA24 /
// NNUE_NYUGYOKU_ENCODING_ONEHOT、既定は 27 点法・thermometer)、評価関数ファイルの
// ハッシュ値に含まれる。このアーキテクチャのビルドではハッシュ値が一致しない評価関数
// ファイルの読み込みをエラーにする (NNUE_STRICT_HASH_CHECK)。

#ifndef CLASSIC_NNUE_HALFKP_NYUGYOKU_256X2_32_32_H_INCLUDED
#define CLASSIC_NNUE_HALFKP_NYUGYOKU_256X2_32_32_H_INCLUDED

#include "../features/feature_set.h"
#include "../features/half_kp.h"

#include "../layers/affine_transform.h"
#include "../layers/affine_transform_nyugyoku_scalar.h"
#include "../layers/clipped_relu.h"

// L1 に入玉スカラーの寄与を足す (evaluate_nnue.cpp の TransformInput が参照する)。
#define NNUE_HAS_NYUGYOKU_SCALAR_INPUT
// 評価関数ファイルのハッシュ値の不一致を警告でなくエラーにする。
#define NNUE_STRICT_HASH_CHECK

namespace YaneuraOu {
namespace Eval::NNUE {

// Input features used in evaluation function
// 評価関数で用いる入力特徴量
using RawFeatures = Features::FeatureSet<Features::HalfKP<Features::Side::kFriend>>;

// Number of input feature dimensions after conversion
// 変換後の入力特徴量の次元数
constexpr IndexType kTransformedFeatureDimensions = 256;

namespace Layers {

// Define network structure
// ネットワーク構造の定義
// L1 は評価関数ファイル上 AffineTransformSparseInput[32<-608] (FT 出力 512 + 入玉スカラー 96)。
// 推論では入玉スカラーの部分を表引きで足す (AffineTransformNyugyokuScalar)。
using NyugyokuL1   = AffineTransformNyugyokuScalar<kTransformedFeatureDimensions * 2, 32>;
using HiddenLayer1 = ClippedReLU<NyugyokuL1>;
using HiddenLayer2 = ClippedReLU<AffineTransform<HiddenLayer1, 32>>;
using OutputLayer  = AffineTransform<HiddenLayer2, 1>;

}  // namespace Layers

using Network = Layers::OutputLayer;

// Network の中の入玉スカラーつき L1 層。
inline const Layers::NyugyokuL1& NyugyokuL1Of(const Network& network) {
	return network.previous_layer().previous_layer().previous_layer().previous_layer();
}

}  // namespace Eval::NNUE
}  // namespace YaneuraOu

#endif  // CLASSIC_NNUE_HALFKP_NYUGYOKU_256X2_32_32_H_INCLUDED
