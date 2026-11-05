// 入玉スカラー特徴: 入玉宣言の点数勘定・宣言条件までの距離を、L1 に直接連結する
// 少数の密な値として表す。
//
// 出力は kDimensions (= 96) 個の uint8_t (0〜127)。並びは
// [手番側の玉のブロック 44 | 相手側の玉のブロック 44 | 0 × 8] で、各ブロックは
// その玉の持ち主から見た量。値の尺度は ClippedReLU 後の FT 出力 (0〜127) と同じ。
//
// ルール表・区切り表・ブロック配置は学習器 tatara の
// crates/shogi-features/src/nyugyoku_scalar.rs と同じ定義。両者の一致は
// eval/nnue/features/testdata/nyugyoku_scalar_golden.tsv (tatara 側にも同一内容の
// コピーがある) を読む UnitTest で検査する。
//
// 宣言条件の判定は Position::DeclarationWin() の条件 (b)(c)(d) に合わせる。
// 手番・王手・持ち時間の条件は含めない (両玉について計算するため)。
// 必要点数はエンジンの EnteringKingRule 設定ではなく、ネットに固定したルールで決める。

#ifndef CLASSIC_NNUE_FEATURES_NYUGYOKU_SCALAR_H
#define CLASSIC_NNUE_FEATURES_NYUGYOKU_SCALAR_H

#include "../../../config.h"

#if defined(EVAL_NNUE)

#include <cstdint>
#include <cstring>
#include <string>

// position.h は NNUE のアーキテクチャ定義を経由してこのヘッダを include するので、
// ここでは include せず Position を前方宣言する。
#include "../../../types.h"
#include "../../../misc.h"

#if defined(USE_AVX2)
#include <immintrin.h>
#endif

namespace YaneuraOu {

class Position;

namespace Test {
class UnitTester;
}

namespace Eval::NNUE::Features::NyugyokuScalar {

// 出力ベクトルの次元 (2 玉分のブロック＋32 の倍数へのパディング)。
constexpr int kDimensions = 96;

// 玉 1 つ分のブロックの次元。
constexpr int kBlockDimensions = 44;

// thermometer 符号化で 1 を表す値。
constexpr std::uint8_t kOne = 127;

// 宣言点数の残り r に対する thermometer の区切り (r <= t を 1 にする)。
constexpr int kNumRemainingPointThresholds = 10;
constexpr int kRemainingPointThresholds[kNumRemainingPointThresholds] = {
  0, 1, 2, 3, 5, 8, 12, 16, 20, 28};

// 入玉宣言の点数ルール。ネット側に固定する。
enum class DeclarationRule : std::uint8_t {
	Csa27,  // CSA ルール (27 点法、平手)。先手 28 点・後手 27 点以上。
	Csa24,  // 24 点法。先後とも 31 点以上 (引き分け帯は特徴化しない)。
};

// ブロック内の値の符号化。
enum class ScalarEncoding : std::uint8_t {
	Thermometer,  // 段数・枚数・残り点数を thermometer で表す。
	OneHot,       // 玉の段を one-hot、枚数・点数を正規化した 1 値で表す。
};

// owner が宣言に必要な点数。
constexpr int required_points(DeclarationRule rule, Color owner) {
	return rule == DeclarationRule::Csa24 ? 31 : (owner == BLACK ? 28 : 27);
}

// golden ファイル・設定で使う名前。
const char* canonical_name(DeclarationRule rule);
const char* canonical_name(ScalarEncoding encoding);

// canonical_name() の逆引き。未知の名前なら false。
bool from_canonical_name(const std::string& name, DeclarationRule* rule);
bool from_canonical_name(const std::string& name, ScalarEncoding* encoding);

// 玉 1 つについての入玉宣言の状態 (すべて玉の持ち主から見た量)。
struct KingDeclarationState {
	int  advance;             // 自陣最下段からの前進段数 (0〜8)
	bool king_in_enemy_camp;  // 玉が敵陣三段目以内
	int  pieces_in_camp;      // 敵陣内の玉以外の自駒の枚数
	int  declaration_points;  // 敵陣内の玉以外の自駒＋持ち駒の点数
	int  potential_points;    // 盤上の玉以外の全自駒＋持ち駒の点数
	int  hand_points;         // 持ち駒の点数
	int  required_points;     // 宣言に必要な点数

	int remaining_points() const {
		return declaration_points >= required_points ? 0
		                                             : required_points - declaration_points;
	}
	int remaining_potential_points() const {
		return potential_points >= required_points ? 0 : required_points - potential_points;
	}
	// 手番・王手・持ち時間を除いた宣言条件をすべて満たす。
	bool declarable() const {
		return king_in_enemy_camp && pieces_in_camp >= 10
		    && declaration_points >= required_points;
	}
};

// pos の owner の玉について入玉宣言の状態を求める。
KingDeclarationState king_declaration_state(const Position& pos, Color owner,
                                            DeclarationRule rule);

// ----------------------------------------------------------------------------
//  L1 への寄与を表引きで求めるためのブロックの分解
// ----------------------------------------------------------------------------
//
// 玉 1 つ分のブロックの列を、互いに重ならない次の 2 種類に分ける。
//   - 表引きの field: 玉の状態の小さな整数 (前進段数など) だけで決まる列の組。値ごとの
//     L1 への寄与を前計算して表から引く
//   - 線形の列: 量子化した比率など、値をそのまま L1 の weight に掛ける 1 列
// ブロックは「各 field の列」と「線形の列」の和なので、表引きと積和の合計は
// 96 次元のスカラーを作って L1 に掛けた値と整数として一致する。write_king_block も
// この分解で書くので、golden テストがこの分解の正しさも検査する。

// 残り点数の field の行数。区切りの最大値 + 1 以上の残りは全区切りで 0 なので 1 行にまとめる。
constexpr int kRemainingRows = kRemainingPointThresholds[kNumRemainingPointThresholds - 1] + 2;

constexpr int kMaxLookupFields  = 4;
constexpr int kMaxLinearColumns = 4;

struct L1Layout {
	int num_lookup;                          // 表引きの field の数
	int lookup_rows[kMaxLookupFields];       // field の取り得る値の数
	int num_linear;                          // 線形の列の数
	int linear_columns[kMaxLinearColumns];   // 線形の列のブロック内の位置
};

// field 0 は「前進段数 d (0〜8)・玉の持ち主が先手か・宣言条件の成立」を
// d + 9 × 先手 + 18 × 成立 (36 行) にまとめ、前進段数・玉が敵陣 (d >= 6 と同値)・先手・
// 宣言条件の成立の列を 1 回の表引きで足す。
// thermometer: 前進段数＋敵陣＋先手＋成立 (36) / 敵陣内の駒数 (11) / 宣言点数の残り (30) /
//   潜在宣言点数の残り (30)。線形の列は持ち駒点数の比率 (39)。
// one-hot: 前進段数＋敵陣＋先手＋成立 (36)。線形の列は駒数・宣言点数・残り・持ち駒点数の
//   比率 (10〜13)。
constexpr int kKingRows = 9 * 2 * 2;
constexpr L1Layout l1_layout(ScalarEncoding encoding) {
	return encoding == ScalarEncoding::Thermometer
	       ? L1Layout{4, {kKingRows, 11, kRemainingRows, kRemainingRows}, 1, {39, 0, 0, 0}}
	       : L1Layout{1, {kKingRows, 0, 0, 0}, 4, {10, 11, 12, 13}};
}

// 玉 1 つ分の、表引きの field の値と線形の列の値。
struct L1Inputs {
	std::uint8_t lookup[kMaxLookupFields];
	std::uint8_t linear[kMaxLinearColumns];
};

L1Inputs l1_inputs(const KingDeclarationState& state, Color owner, ScalarEncoding encoding);

// pos の手番側 (out[0]) と相手側 (out[1]) の玉の L1Inputs をまとめて求める (評価用)。
// l1_inputs(king_declaration_state(...)) を両玉に使ったのと同じ値になる。
void l1_inputs_both(const Position& pos, DeclarationRule rule, ScalarEncoding encoding,
                    L1Inputs out[2]);

// pos の owner の玉の L1Inputs (評価用)。
L1Inputs l1_inputs_for(const Position& pos, Color owner, DeclarationRule rule,
                       ScalarEncoding encoding);

// field = value のときにブロックへ書かれる列だけを block に書く (他の列は変えない)。
void write_lookup_field(ScalarEncoding encoding, int field, int value,
                        std::uint8_t block[kBlockDimensions]);

// 入玉スカラー 96 次元に掛かる L1 の weight (OutputDimensions × 96) から、玉の側
// (0: 手番側、1: 相手側) ごとの寄与表を前計算する。符号化はネットに固定なので型に持たせ、
// field の数・行の位置をコンパイル時の定数にする (評価のたびの表引きのループが展開される)。
//
// 表引きの field の列は 0 か kOne のフラグだけなので、表には「立つ列の weight の和」を
// int16 で持ち、全 field を足した後で kOne を掛ける (行が 32 出力で 64 バイト = 1 キャッシュ
// ライン)。両玉の全列が立って weight が全部 -128 でも和は int16 に収まる。
template<int OutputDimensions, ScalarEncoding Encoding>
struct L1ContributionTable {
	static_assert(2 * kBlockDimensions * 128 <= 32767,
	              "the sum of all flag-column weights must fit in int16");

	static constexpr L1Layout kLayout = l1_layout(Encoding);

	// field f の先頭の行
	static constexpr int row_offset(int f) {
		int offset = 0;
		for (int i = 0; i < f; ++i)
			offset += kLayout.lookup_rows[i];
		return offset;
	}
	static constexpr int kRows = row_offset(kLayout.num_lookup);

	// [側][表引きの行][出力]: 立つ列の weight の和
	alignas(64) std::int16_t lookup[2][kRows][OutputDimensions];
	// [側][線形の列][出力]
	alignas(64) std::int32_t linear[2][kLayout.num_linear][OutputDimensions];

	// weights[o * stride + k] が出力 o・スカラー k の weight。
	void build(const std::int8_t* weights, int stride) {
		for (int side = 0; side < 2; ++side)
		{
			const int base = side * kBlockDimensions;
			for (int f = 0; f < kLayout.num_lookup; ++f)
				for (int v = 0; v < kLayout.lookup_rows[f]; ++v)
				{
					std::uint8_t cells[kBlockDimensions] = {};
					write_lookup_field(Encoding, f, v, cells);
					for (int o = 0; o < OutputDimensions; ++o)
					{
						int sum = 0;
						for (int c = 0; c < kBlockDimensions; ++c)
						{
							ASSERT_LV1(cells[c] == 0 || cells[c] == kOne);
							if (cells[c] == kOne)
								sum += weights[o * stride + base + c];
						}
						lookup[side][row_offset(f) + v][o] = std::int16_t(sum);
					}
				}
			for (int i = 0; i < kLayout.num_linear; ++i)
				for (int o = 0; o < OutputDimensions; ++o)
					linear[side][i][o] = weights[o * stride + base + kLayout.linear_columns[i]];
		}
	}

	// add() が読む行を L1 キャッシュへ先読みする。表は探索中に FT の weight などに押し出されて
	// いることが多いので、FT の差分更新の前に呼んで、行の読み込みをその計算と重ねる。
	void prefetch(const L1Inputs& stm, const L1Inputs& nstm) const {
		const L1Inputs* inputs[2] = {&stm, &nstm};
		for (int side = 0; side < 2; ++side)
		{
			for (int f = 0; f < kLayout.num_lookup; ++f)
				prefetch_row(lookup[side][row_offset(f) + inputs[side]->lookup[f]]);
			for (int i = 0; i < kLayout.num_linear; ++i)
				prefetch_row(linear[side][i]);
		}
	}

	// 手番側 (stm) と相手側 (nstm) の玉の寄与を out に足す。
	void add(const L1Inputs& stm, const L1Inputs& nstm, std::int32_t* out) const {
		const L1Inputs* inputs[2] = {&stm, &nstm};
#if defined(USE_AVX2)
		// 出力 32 個を AVX2 のレジスタに置いたまま足す。表の行は int16 のレジスタ 2 本に
		// 足してから int32 に広げて kOne を掛け、線形の列は int32 のレジスタ 4 本に足す。
		static_assert(OutputDimensions == 32, "the AVX2 path keeps 32 outputs in registers");
		const auto out_v = reinterpret_cast<__m256i*>(out);
		__m256i    s0 = _mm256_setzero_si256(), s1 = _mm256_setzero_si256();
		for (int side = 0; side < 2; ++side)
			for (int f = 0; f < kLayout.num_lookup; ++f)
			{
				const auto r = reinterpret_cast<const __m256i*>(
				  lookup[side][row_offset(f) + inputs[side]->lookup[f]]);
				s0 = _mm256_add_epi16(s0, _mm256_loadu_si256(r + 0));
				s1 = _mm256_add_epi16(s1, _mm256_loadu_si256(r + 1));
			}
		// x * kOne (= x * 128 - x)
		const auto times_one = [](__m128i x16) {
			const __m256i x = _mm256_cvtepi16_epi32(x16);
			return _mm256_sub_epi32(_mm256_slli_epi32(x, 7), x);
		};
		static_assert(kOne == 127);
		__m256i a0 = _mm256_add_epi32(_mm256_loadu_si256(out_v + 0),
		                              times_one(_mm256_castsi256_si128(s0)));
		__m256i a1 = _mm256_add_epi32(_mm256_loadu_si256(out_v + 1),
		                              times_one(_mm256_extracti128_si256(s0, 1)));
		__m256i a2 = _mm256_add_epi32(_mm256_loadu_si256(out_v + 2),
		                              times_one(_mm256_castsi256_si128(s1)));
		__m256i a3 = _mm256_add_epi32(_mm256_loadu_si256(out_v + 3),
		                              times_one(_mm256_extracti128_si256(s1, 1)));
		for (int side = 0; side < 2; ++side)
			for (int i = 0; i < kLayout.num_linear; ++i)
			{
				const __m256i x = _mm256_set1_epi32(inputs[side]->linear[i]);
				const auto    w = reinterpret_cast<const __m256i*>(linear[side][i]);
				a0 = _mm256_add_epi32(a0, _mm256_mullo_epi32(_mm256_loadu_si256(w + 0), x));
				a1 = _mm256_add_epi32(a1, _mm256_mullo_epi32(_mm256_loadu_si256(w + 1), x));
				a2 = _mm256_add_epi32(a2, _mm256_mullo_epi32(_mm256_loadu_si256(w + 2), x));
				a3 = _mm256_add_epi32(a3, _mm256_mullo_epi32(_mm256_loadu_si256(w + 3), x));
			}
		_mm256_storeu_si256(out_v + 0, a0);
		_mm256_storeu_si256(out_v + 1, a1);
		_mm256_storeu_si256(out_v + 2, a2);
		_mm256_storeu_si256(out_v + 3, a3);
#else
		// 局所配列に足してから書き戻す (表の行と out が重ならないことをコンパイラに示す)。
		std::int32_t acc[OutputDimensions];
		std::memcpy(acc, out, sizeof(acc));
		for (int side = 0; side < 2; ++side)
		{
			const L1Inputs& in = *inputs[side];
			for (int f = 0; f < kLayout.num_lookup; ++f)
			{
				const std::int16_t* row = lookup[side][row_offset(f) + in.lookup[f]];
				for (int o = 0; o < OutputDimensions; ++o)
					acc[o] += kOne * row[o];
			}
			for (int i = 0; i < kLayout.num_linear; ++i)
			{
				const std::int32_t  x = in.linear[i];
				const std::int32_t* w = linear[side][i];
				for (int o = 0; o < OutputDimensions; ++o)
					acc[o] += w[o] * x;
			}
		}
		std::memcpy(out, acc, sizeof(acc));
#endif
	}

   private:
	template<typename T>
	static void prefetch_row(const T* row) {
		constexpr std::size_t kLine = 64;
		for (std::size_t b = 0; b < sizeof(T) * OutputDimensions; b += kLine)
			YaneuraOu::prefetch(reinterpret_cast<const char*>(row) + b);
	}
};

// 玉 1 つ分のブロック (kBlockDimensions 個) を out に書く。
void write_king_block(const KingDeclarationState& state, Color owner, ScalarEncoding encoding,
                      std::uint8_t out[kBlockDimensions]);

// pos の入玉スカラーを out[0..kDimensions) に書く。
void compute(const Position& pos, DeclarationRule rule, ScalarEncoding encoding,
             std::uint8_t out[kDimensions]);

// golden ファイルとの突合と、DeclarationWin() との整合を検査する。
// golden ファイルの場所は "unittest nyugyoku_scalar_golden <path>" で指定でき、
// 省略時は実行ファイルのフォルダからの相対パス
// eval/nnue/features/testdata/nyugyoku_scalar_golden.tsv を読む。
void UnitTest(Test::UnitTester& tester, IEngine& engine);

}  // namespace Eval::NNUE::Features::NyugyokuScalar
}  // namespace YaneuraOu

#endif  // defined(EVAL_NNUE)

#endif
