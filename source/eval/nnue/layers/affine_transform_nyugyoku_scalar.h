// NNUE評価関数の L1 層: 特徴変換器 (FT) の出力 + 入玉スカラーを入力とするアフィン変換
//
// 評価関数ファイル上は AffineTransformSparseInput<NyugyokuScalarInputSlice<FT>, OUT> と同じ
// (bias OUT 個、weight OUT × (FT + 96) 行優先、ハッシュ値・構造文字列も同じ) で、学習器の
// export をそのまま読む。
//
// 推論では入玉スカラー 96 次元を作らない。
//   - FT 出力の部分は AffineTransformSparseInput で計算する
//   - 入玉スカラーの部分は、読み込み時に前計算した「玉の状態 → OUT 出力への寄与」の表
//     (NyugyokuScalar::L1ContributionTable) を引いて足す
// 入力 buffer の FT 出力の直後には、入玉スカラーの代わりに玉 2 つ分の L1Inputs
// (表引きの index と線形の列の値) を evaluate_nnue.cpp の TransformInput が書く。
// アフィン変換は ClippedReLU の前の整数の和なので、足す順序を変えても結果は
// 96 次元を作って掛けた場合と一致する。
// 表の行は探索中に L1 キャッシュから追い出されていることが多いので、TransformInput が
// FT の差分更新の前に PrefetchTable で引く行を先読みしておく。

#ifndef CLASSIC_NNUE_LAYERS_AFFINE_TRANSFORM_NYUGYOKU_SCALAR_H_INCLUDED
#define CLASSIC_NNUE_LAYERS_AFFINE_TRANSFORM_NYUGYOKU_SCALAR_H_INCLUDED

#include "../../../config.h"

#if defined(EVAL_NNUE)

#include <cstring>
#include <sstream>
#include <string>

#include "../nnue_common.h"
#include "../features/nyugyoku_scalar.h"
#include "affine_transform_sparse_input.h"
#include "input_slice.h"
#include "nyugyoku_scalar_input_slice.h"

namespace YaneuraOu {
namespace Eval::NNUE::Layers {

template<IndexType FtDimensions, IndexType OutputDimensions>
class AffineTransformNyugyokuScalar {
   public:
	using InputType  = TransformedFeatureType;
	using OutputType = std::int32_t;

	// 評価関数ファイル上の入力層 (ハッシュ値・構造文字列の計算に使う)
	using FileInputLayer = NyugyokuScalarInputSlice<FtDimensions>;
	// FT 出力の部分を計算する層
	using FtAffine = AffineTransformSparseInput<InputSlice<FtDimensions>, OutputDimensions>;
	using Table    = Features::NyugyokuScalar::L1ContributionTable<OutputDimensions, kNyugyokuEncoding>;

	static constexpr IndexType kFtDimensions     = FtDimensions;
	static constexpr IndexType kScalarDimensions = FileInputLayer::kScalarDimensions;
	static constexpr IndexType kInputDimensions  = FileInputLayer::kOutputDimensions;
	static constexpr IndexType kOutputDimensions = OutputDimensions;
	static constexpr IndexType kPaddedInputDimensions =
	  CeilToMultiple<IndexType>(kInputDimensions, kMaxSimdWidth);

	// 入力 buffer の FT 出力の直後に置く、玉 2 つ分 (手番側・相手側) の L1Inputs
	static constexpr std::size_t kInputStateBytes = 2 * sizeof(Features::NyugyokuScalar::L1Inputs);
	static_assert(kInputStateBytes <= kScalarDimensions,
	              "the packed king states must fit where the scalars used to be");

	static constexpr std::size_t kBufferSize = FtAffine::kBufferSize;

	// ハッシュ値・構造文字列は AffineTransformSparseInput<FileInputLayer, OUT> と同じ
	static constexpr std::uint32_t GetHashValue() {
		std::uint32_t hash_value = 0xCC03DAE4u;
		hash_value += kOutputDimensions;
		hash_value ^= FileInputLayer::GetHashValue() >> 1;
		hash_value ^= FileInputLayer::GetHashValue() << 31;
		return hash_value;
	}

	static std::string GetStructureString() {
		return "AffineTransformSparseInput[" + std::to_string(kOutputDimensions) + "<-"
		     + std::to_string(kInputDimensions) + "](" + FileInputLayer::GetStructureString()
		     + ")";
	}

	Tools::Result ReadParameters(std::istream& stream) {
		for (IndexType o = 0; o < kOutputDimensions; ++o)
			biases_[o] = read_little_endian<std::int32_t>(stream);
		for (IndexType i = 0; i < kOutputDimensions * kPaddedInputDimensions; ++i)
			weights_[i] = read_little_endian<std::int8_t>(stream);
		if (stream.fail())
			return Tools::ResultCode::FileReadError;

		// FT 出力の部分 (各行の先頭 FtDimensions 列) を FtAffine の形式で読ませる。
		std::string ft_bytes;
		ft_bytes.reserve(sizeof(biases_) + kOutputDimensions * FtAffine::kPaddedInputDimensions);
		ft_bytes.append(reinterpret_cast<const char*>(biases_), sizeof(biases_));
		static_assert(FtAffine::kPaddedInputDimensions == kFtDimensions,
		              "FT output width must already be a multiple of the SIMD width");
		for (IndexType o = 0; o < kOutputDimensions; ++o)
			ft_bytes.append(reinterpret_cast<const char*>(&weights_[o * kPaddedInputDimensions]),
			                kFtDimensions);
		std::istringstream ft_stream(ft_bytes);
		const Tools::Result result = ft_affine_.ReadParameters(ft_stream);
		if (result.is_not_ok())
			return result;

		table_.build(&weights_[kFtDimensions], int(kPaddedInputDimensions));
		return Tools::ResultCode::Ok;
	}

	// Propagate で引く表の行を L1 キャッシュへ先読みする。
	void PrefetchTable(const Features::NyugyokuScalar::L1Inputs& stm,
	                   const Features::NyugyokuScalar::L1Inputs& nstm) const {
		table_.prefetch(stm, nstm);
	}

	bool WriteParameters(std::ostream& stream) const {
		stream.write(reinterpret_cast<const char*>(biases_), sizeof(biases_));
		stream.write(reinterpret_cast<const char*>(weights_), sizeof(weights_));
		return !stream.fail();
	}

	// transformed_features: FT 出力 kFtDimensions バイトの直後に L1Inputs × 2
	const OutputType* Propagate(const TransformedFeatureType* transformed_features,
	                            char* buffer) const {
		Features::NyugyokuScalar::L1Inputs states[2];
		std::memcpy(states, transformed_features + kFtDimensions, sizeof(states));
		ft_affine_.Propagate(transformed_features, buffer);
		auto output = reinterpret_cast<OutputType*>(buffer);
		table_.add(states[0], states[1], output);
		return output;
	}

	// 前計算した表のサイズ (バイト)
	static constexpr std::size_t TableBytes() { return sizeof(Table); }

   private:
	FtAffine     ft_affine_;
	Table        table_;
	// 評価関数ファイルのまま (WriteParameters で書き戻す)
	std::int32_t biases_[kOutputDimensions];
	std::int8_t  weights_[kOutputDimensions * kPaddedInputDimensions];
};

}  // namespace Eval::NNUE::Layers
}  // namespace YaneuraOu

#endif  // defined(EVAL_NNUE)

#endif  // CLASSIC_NNUE_LAYERS_AFFINE_TRANSFORM_NYUGYOKU_SCALAR_H_INCLUDED
