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
#include <string>

// position.h は NNUE のアーキテクチャ定義を経由してこのヘッダを include するので、
// ここでは include せず Position を前方宣言する。
#include "../../../types.h"

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
