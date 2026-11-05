// NNUE評価関数の入力層: 特徴変換器 (FT) の出力の後ろに入玉スカラー 96 個を連結した L1 入力
//
// 層自体は入力 buffer をそのまま渡すだけで、入玉スカラーは呼び出し側
// (evaluate_nnue.cpp の TransformInput) が FT 出力の直後に書く。この層は
//   - L1 の入力次元 (FT 出力 + 96)
//   - 評価関数ファイルのハッシュ値 (ネットに固定した入玉ルール・符号化を含む)
//   - 構造文字列
// を定める。ハッシュ値と構造文字列は学習器 tatara の
// crates/nnue-format/src/simple_weights.rs (fc_hash / arch_identity) と同じ値になる。

#ifndef CLASSIC_NNUE_LAYERS_NYUGYOKU_SCALAR_INPUT_SLICE_H_INCLUDED
#define CLASSIC_NNUE_LAYERS_NYUGYOKU_SCALAR_INPUT_SLICE_H_INCLUDED

#include "../../../config.h"

#if defined(EVAL_NNUE)

#include <cstdint>
#include <string>

#include "../nnue_common.h"
#include "../features/nyugyoku_scalar.h"

namespace YaneuraOu {
namespace Eval::NNUE::Layers {

// ネットに固定する入玉ルールと符号化 (ビルドフラグで選ぶ。既定は 27 点法・thermometer)。
#if defined(NNUE_NYUGYOKU_RULE_CSA24)
constexpr Features::NyugyokuScalar::DeclarationRule kNyugyokuRule =
  Features::NyugyokuScalar::DeclarationRule::Csa24;
	#define NNUE_NYUGYOKU_RULE_NAME "csa24"
#else
constexpr Features::NyugyokuScalar::DeclarationRule kNyugyokuRule =
  Features::NyugyokuScalar::DeclarationRule::Csa27;
	#define NNUE_NYUGYOKU_RULE_NAME "csa27"
#endif

#if defined(NNUE_NYUGYOKU_ENCODING_ONEHOT)
constexpr Features::NyugyokuScalar::ScalarEncoding kNyugyokuEncoding =
  Features::NyugyokuScalar::ScalarEncoding::OneHot;
	#define NNUE_NYUGYOKU_ENCODING_NAME "onehot"
#else
constexpr Features::NyugyokuScalar::ScalarEncoding kNyugyokuEncoding =
  Features::NyugyokuScalar::ScalarEncoding::Thermometer;
	#define NNUE_NYUGYOKU_ENCODING_NAME "thermometer"
#endif

static_assert(Features::NyugyokuScalar::kDimensions == 96,
              "the structure string below spells out the 96 scalar dimensions");

// 入玉スカラーの密入力を表すトークン。ハッシュ値の計算と構造文字列の両方に使う。
inline constexpr char kNyugyokuScalarToken[] =
  "NyugyokuScalar(" NNUE_NYUGYOKU_RULE_NAME "," NNUE_NYUGYOKU_ENCODING_NAME ")[96]";

// FNV-1a 32bit。
constexpr std::uint32_t Fnv1a32(const char* s) {
	std::uint32_t h = 0x811C9DC5u;
	for (; *s != '\0'; ++s)
	{
		h ^= static_cast<unsigned char>(*s);
		h *= 0x01000193u;
	}
	return h;
}

template<IndexType FtDimensions>
class NyugyokuScalarInputSlice {
   public:
	using OutputType = TransformedFeatureType;

	// FT 出力 (手番側 + 相手側) の次元と、その後ろに続く入玉スカラーの次元
	static constexpr IndexType kFtDimensions     = FtDimensions;
	static constexpr IndexType kScalarDimensions = Features::NyugyokuScalar::kDimensions;
	static constexpr IndexType kOutputDimensions = kFtDimensions + kScalarDimensions;

	// 入力層からこの層までで使用する順伝播用バッファのサイズ
	static constexpr std::size_t kBufferSize = 0;

	// 評価関数ファイルに埋め込むハッシュ値。InputSlice の値に入玉スカラーのトークンの
	// FNV-1a を XOR し、ルール・符号化の違うネットを区別する。
	static constexpr std::uint32_t GetHashValue() {
		return 0xEC42E90Du ^ kOutputDimensions ^ Fnv1a32(kNyugyokuScalarToken);
	}

	// 入力層からこの層までの構造を表す文字列
	static std::string GetStructureString() {
		return "InputSlice[" + std::to_string(kOutputDimensions) + "(0:"
		     + std::to_string(kFtDimensions) + ")+" + kNyugyokuScalarToken + "("
		     + std::to_string(kFtDimensions) + ":" + std::to_string(kOutputDimensions) + ")]";
	}

	Tools::Result ReadParameters(std::istream& /*stream*/) { return Tools::ResultCode::Ok; }

	bool WriteParameters(std::ostream& /*stream*/) const { return true; }

	// 順伝播。transformed_features は kOutputDimensions バイトで、FT 出力の後ろに
	// 入玉スカラーが書かれていること。
	const OutputType* Propagate(const TransformedFeatureType* transformed_features,
	                            char* /*buffer*/) const {
		return transformed_features;
	}
};

}  // namespace Eval::NNUE::Layers
}  // namespace YaneuraOu

#endif  // defined(EVAL_NNUE)

#endif  // CLASSIC_NNUE_LAYERS_NYUGYOKU_SCALAR_INPUT_SLICE_H_INCLUDED
