// 入玉スカラー特徴の計算と UnitTest

#include "nyugyoku_scalar.h"

#if defined(EVAL_NNUE)

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <type_traits>
#include <vector>

#include "../../../misc.h"
#include "../../../position.h"
#include "../../../testcmd/unit_test.h"

namespace YaneuraOu {
namespace Eval::NNUE::Features::NyugyokuScalar {

const char* canonical_name(DeclarationRule rule) {
	return rule == DeclarationRule::Csa27 ? "csa27" : "csa24";
}

const char* canonical_name(ScalarEncoding encoding) {
	return encoding == ScalarEncoding::Thermometer ? "thermometer" : "onehot";
}

bool from_canonical_name(const std::string& name, DeclarationRule* rule) {
	for (auto r : {DeclarationRule::Csa27, DeclarationRule::Csa24})
		if (name == canonical_name(r))
		{
			*rule = r;
			return true;
		}
	return false;
}

bool from_canonical_name(const std::string& name, ScalarEncoding* encoding) {
	for (auto e : {ScalarEncoding::Thermometer, ScalarEncoding::OneHot})
		if (name == canonical_name(e))
		{
			*encoding = e;
			return true;
		}
	return false;
}

namespace {

// 持ち駒の点数 (大駒 5 点・小駒 1 点)。
int hand_points(Hand h) {
	return hand_count(h, PAWN) + hand_count(h, LANCE) + hand_count(h, KNIGHT)
	     + hand_count(h, SILVER) + hand_count(h, GOLD)
	     + (hand_count(h, BISHOP) + hand_count(h, ROOK)) * 5;
}

// round(127 * num / den) (0.5 は切り上げ)。0 <= num <= den、den > 0 を caller が保証する。
// tatara 側と同じ整数式で計算し、浮動小数の丸め差を避ける。評価のたびに呼ばれるので、
// 分母が使われる範囲 (必要点 27・28・31 と 10) に収まる値は前計算した表から引く。
constexpr int kScaleTableSize = 32;
struct ScaleTable {
	std::uint8_t value[kScaleTableSize][kScaleTableSize];
	constexpr ScaleTable() : value{} {
		for (int den = 1; den < kScaleTableSize; ++den)
			for (int num = 0; num <= den; ++num)
				value[den][num] = std::uint8_t((254 * num + den) / (2 * den));
	}
};
constexpr ScaleTable kScaleTable;

std::uint8_t scale_to_u8(int num, int den) {
	if (den < kScaleTableSize)
		return kScaleTable.value[den][num];
	return std::uint8_t((254 * num + den) / (2 * den));
}

std::uint8_t flag(bool b) { return b ? kOne : 0; }

// 玉 1 つ分のブロックを out[0..kBlockDimensions) に書く。配置は tatara 側の
// write_block と同じ:
//
// thermometer: [0,8) 前進段数 d>=1..8 / 8 玉が敵陣 / [9,19) 敵陣内の駒数 n>=1..10 /
//   [19,29) 宣言点数の残り r<=t / [29,39) 潜在宣言点数の残り / 39 持ち駒点数/必要点 /
//   40 宣言条件の成立 / 41 先手 / [42,44) 予約 (0)
// one-hot: [0,9) 前進段数 d==0..8 / 9 玉が敵陣 / 10 駒数/10 / 11 宣言点数/必要点 /
//   12 残り/必要点 / 13 持ち駒点数/必要点 / 14 宣言条件の成立 / 15 先手 / [16,44) 未使用
void write_block(const KingDeclarationState& s, Color owner, ScalarEncoding encoding,
                 std::uint8_t* out) {
	std::fill(out, out + kBlockDimensions, std::uint8_t(0));
	const L1Layout layout = l1_layout(encoding);
	const L1Inputs inputs = l1_inputs(s, owner, encoding);
	for (int f = 0; f < layout.num_lookup; ++f)
		write_lookup_field(encoding, f, inputs.lookup[f], out);
	for (int i = 0; i < layout.num_linear; ++i)
		out[layout.linear_columns[i]] = inputs.linear[i];
}

}  // namespace

namespace {

inline L1Inputs l1_inputs_inline(const KingDeclarationState& s, Color owner,
                                 ScalarEncoding encoding) {
	L1Inputs     in{};
	const int    need       = s.required_points;
	const std::uint8_t hand_ratio = scale_to_u8(std::min(s.hand_points, need), need);
	// field 0: 前進段数 + 9 × 先手 + 18 × 宣言条件の成立 (玉が敵陣かは前進段数 >= 6 と
	// 同値なので同じ表に入る)
	in.lookup[0] = std::uint8_t(s.advance + 9 * (owner == BLACK) + 18 * s.declarable());
	if (encoding == ScalarEncoding::Thermometer)
	{
		in.lookup[1] = std::uint8_t(std::min(s.pieces_in_camp, 10));
		in.lookup[2] = std::uint8_t(std::min(s.remaining_points(), kRemainingRows - 1));
		in.lookup[3] = std::uint8_t(std::min(s.remaining_potential_points(), kRemainingRows - 1));
		in.linear[0] = hand_ratio;
	}
	else
	{
		in.linear[0] = scale_to_u8(std::min(s.pieces_in_camp, 10), 10);
		in.linear[1] = scale_to_u8(std::min(s.declaration_points, need), need);
		in.linear[2] = scale_to_u8(s.remaining_points(), need);
		in.linear[3] = hand_ratio;
	}
	return in;
}

// 玉 1 つ分の状態。big は盤上の大駒 (角・馬・飛・竜) の bitboard (両者分)。
inline KingDeclarationState king_state_inline(const Position& pos, Color owner,
                                              DeclarationRule rule, const Bitboard& big) {
	const Square   ksq          = pos.square<KING>(owner);
	const Bitboard camp         = enemy_field(owner);
	const Bitboard own          = pos.pieces(owner);
	const Bitboard own_big      = own & big;
	const bool     king_in_camp = bool(camp & ksq);

	// 玉を含めて数え、玉の分を引く。大駒は枚数×4 を足して 5 点にする。
	const int board_pieces = own.pop_count() - 1;
	const int board_big    = own_big.pop_count();
	const int camp_pieces  = (own & camp).pop_count() - (king_in_camp ? 1 : 0);
	const int camp_big     = (own_big & camp).pop_count();
	const int hand         = hand_points(pos.hand_of(owner));
	const int rank         = int(rank_of(ksq));

	KingDeclarationState s;
	s.advance            = owner == BLACK ? 8 - rank : rank;
	s.king_in_enemy_camp = king_in_camp;
	s.pieces_in_camp     = camp_pieces;
	s.declaration_points = camp_pieces + camp_big * 4 + hand;
	s.potential_points   = board_pieces + board_big * 4 + hand;
	s.hand_points        = hand;
	s.required_points    = required_points(rule, owner);
	return s;
}

}  // namespace

L1Inputs l1_inputs(const KingDeclarationState& s, Color owner, ScalarEncoding encoding) {
	return l1_inputs_inline(s, owner, encoding);
}

L1Inputs l1_inputs_for(const Position& pos, Color owner, DeclarationRule rule,
                       ScalarEncoding encoding) {
	return l1_inputs_inline(
	  king_state_inline(pos, owner, rule, pos.pieces(BISHOP_HORSE, ROOK_DRAGON)), owner, encoding);
}

void l1_inputs_both(const Position& pos, DeclarationRule rule, ScalarEncoding encoding,
                    L1Inputs out[2]) {
	const Bitboard big = pos.pieces(BISHOP_HORSE, ROOK_DRAGON);
	const Color    us  = pos.side_to_move();
	out[0] = l1_inputs_inline(king_state_inline(pos, us, rule, big), us, encoding);
	out[1] = l1_inputs_inline(king_state_inline(pos, ~us, rule, big), ~us, encoding);
}

void write_lookup_field(ScalarEncoding encoding, int field, int value,
                        std::uint8_t block[kBlockDimensions]) {
	if (field == 0)
	{
		// 前進段数 d + 9 × 先手 + 18 × 宣言条件の成立。玉が敵陣 (敵陣三段目以内) は d >= 6 と同値。
		const int  d          = value % 9;
		const bool black      = (value / 9) % 2 != 0;
		const bool declarable = value >= 18;
		if (encoding == ScalarEncoding::Thermometer)
		{
			for (int k = 1; k <= 8; ++k)
				block[k - 1] = flag(d >= k);
			block[8]  = flag(d >= 6);
			block[40] = flag(declarable);
			block[41] = flag(black);
		}
		else
		{
			block[d]  = kOne;
			block[9]  = flag(d >= 6);
			block[14] = flag(declarable);
			block[15] = flag(black);
		}
		return;
	}
	if (encoding == ScalarEncoding::Thermometer)
	{
		switch (field)
		{
		case 1:  // 敵陣内の駒数 n >= 1 … n >= 10
			for (int k = 1; k <= 10; ++k)
				block[8 + k] = flag(value >= k);
			break;
		case 2:  // 宣言点数の残り r <= t
		case 3:  // 潜在宣言点数の残り
			for (int i = 0; i < kNumRemainingPointThresholds; ++i)
				block[(field == 2 ? 19 : 29) + i] = flag(value <= kRemainingPointThresholds[i]);
			break;
		}
	}
}

void write_king_block(const KingDeclarationState& state, Color owner, ScalarEncoding encoding,
                      std::uint8_t out[kBlockDimensions]) {
	write_block(state, owner, encoding, out);
}

KingDeclarationState king_declaration_state(const Position& pos, Color owner,
                                            DeclarationRule rule) {
	return king_state_inline(pos, owner, rule, pos.pieces(BISHOP_HORSE, ROOK_DRAGON));
}

void compute(const Position& pos, DeclarationRule rule, ScalarEncoding encoding,
             std::uint8_t out[kDimensions]) {
	const Color us = pos.side_to_move();
	for (int i = 0; i < 2; ++i)
	{
		const Color owner = i == 0 ? us : ~us;
		write_block(king_declaration_state(pos, owner, rule), owner, encoding,
		            out + i * kBlockDimensions);
	}
	std::fill(out + 2 * kBlockDimensions, out + kDimensions, std::uint8_t(0));
}

// ----------------------------------
//      UnitTest
// ----------------------------------

namespace {

// golden ファイルの FNV-1a 32bit (CR を除いたバイト列)。tatara 側の
// crates/shogi-features/tests/nyugyoku_scalar.rs の GOLDEN_FNV1A32 と同じ値。
// golden ファイルは tatara の scripts/gen-nyugyoku-scalar-golden.py で生成する
// (--yaneuraou でこのリポのコピーも書く)。再生成したら、スクリプトが表示する値に
// 両リポの定数をそろえる。
constexpr std::uint32_t kGoldenFnv1a32 = 0xE063FF67u;

std::uint32_t fnv1a32_without_cr(const std::string& bytes) {
	std::uint32_t h = 0x811C9DC5u;
	for (unsigned char c : bytes)
	{
		if (c == '\r')
			continue;
		h ^= c;
		h *= 0x01000193u;
	}
	return h;
}

std::vector<std::string> split(const std::string& s, char sep) {
	std::vector<std::string> out;
	std::string              cur;
	for (char c : s)
	{
		if (c == sep)
		{
			out.push_back(cur);
			cur.clear();
		}
		else if (c != '\r')
			cur += c;
	}
	out.push_back(cur);
	return out;
}

std::string to_string(const std::uint8_t* v, int n) {
	std::string s;
	for (int i = 0; i < n; ++i)
		s += (i ? " " : "") + std::to_string(int(v[i]));
	return s;
}

}  // namespace

void UnitTest(Test::UnitTester& tester, IEngine& /*engine*/) {
	auto section1 = tester.section("NyugyokuScalar");

	std::string path = std::string(tester.options["nyugyoku_scalar_golden"]);
	if (path.empty())
		path = Path::Combine(Directory::GetBinaryFolder(),
		                     "eval/nnue/features/testdata/nyugyoku_scalar_golden.tsv");

	// UnitTester::test() はメッセージが 1 行 (約 70 文字) を超えると扱えないので、
	// パスや SFEN などの長い情報は別の行に出力し、テスト名は短くする。
	std::cout << "golden file: " << path << std::endl;
	std::ifstream file(path, std::ios::binary);
	tester.test("open golden file", bool(file));
	if (!file)
		return;
	const std::string content((std::istreambuf_iterator<char>(file)),
	                          std::istreambuf_iterator<char>());

	tester.test("golden file checksum", fnv1a32_without_cr(content) == kGoldenFnv1a32);

	Position  pos;
	StateInfo si;

	std::vector<std::pair<std::string, std::string>> positions;  // (name, sfen)
	int                                              rows = 0;
	{
		auto section2 = tester.section("Golden");
		std::istringstream lines(content);
		std::string        line;
		while (std::getline(lines, line))
		{
			if (!line.empty() && line.back() == '\r')
				line.pop_back();
			if (line.empty() || line[0] == '#')
				continue;
			const auto cols = split(line, '\t');
			DeclarationRule rule;
			ScalarEncoding  encoding;
			if (cols.size() != 5 || !from_canonical_name(cols[1], &rule)
			    || !from_canonical_name(cols[2], &encoding))
			{
				std::cout << "  malformed row: " << line << std::endl;
				tester.test("parse row", false);
				continue;
			}
			std::vector<int> expected;
			{
				std::istringstream values(cols[4]);
				int                v;
				while (values >> v)
					expected.push_back(v);
			}
			const std::string label = cols[0] + " " + cols[1] + " " + cols[2];
			if (expected.size() != size_t(kDimensions) || pos.set(cols[3], &si).has_value())
			{
				tester.test("parse " + label, false);
				continue;
			}
			std::uint8_t got[kDimensions];
			compute(pos, rule, encoding, got);
			const bool ok = std::equal(expected.begin(), expected.end(), got,
			                           [](int e, std::uint8_t g) { return e == int(g); });
			if (!ok)
				std::cout << "  got: " << to_string(got, kDimensions) << std::endl;
			tester.test(label, ok);
			if (std::none_of(positions.begin(), positions.end(),
			                 [&](const auto& p) { return p.second == cols[3]; }))
				positions.emplace_back(cols[0], cols[3]);
			++rows;
		}
		tester.test("row count", rows == 32);
	}

	// 宣言条件の成立ビットを、エンジン本体の宣言判定 Position::DeclarationWin() と
	// 突き合わせる。DeclarationWin() は手番側について王手なしを前提に判定するので、
	// 各 golden 局面の手番を先後両方に置き換え、王手のかかっていない場合だけ比べる。
	{
		auto section2 = tester.section("DeclarationWin");
		for (const auto& [name, sfen] : positions)
			for (const char* side : {" b ", " w "})
			{
				std::string s   = sfen;
				const auto  pos_side = s.find(" b ") != std::string::npos ? s.find(" b ")
				                                                          : s.find(" w ");
				s.replace(pos_side, 3, side);
				if (pos.set(s, &si).has_value() || pos.checkers())
					continue;
				const Color us = pos.side_to_move();
				for (auto [rule, ekr] : {std::pair{DeclarationRule::Csa27, EKR_27_POINT},
				                         std::pair{DeclarationRule::Csa24, EKR_24_POINT}})
				{
					pos.set_ekr(ekr);
					const bool engine = pos.DeclarationWin() == Move::win();
					const bool ours   = king_declaration_state(pos, us, rule).declarable();
					if (engine != ours)
						std::cout << "  sfen: " << s << std::endl;
					tester.test(name + side + canonical_name(rule), engine == ours);
				}
			}
	}

	// L1 への寄与の表引きが、96 次元を作って weight を掛けた和と一致することを、int8 の
	// 全範囲の乱数の weight と、表の int16 の和が最も大きくなる全 -128 の weight で確かめる。
	{
		auto section2 = tester.section("L1Table");
		constexpr int kOut = 32;
		std::vector<std::int8_t> weights(kOut * kDimensions);
		PRNG prng(20261105);
		for (auto& w : weights)
			w = std::int8_t(int(prng.rand(256)) - 128);
		const auto check_encoding = [&](auto encoding_constant, const char* label) {
			constexpr ScalarEncoding encoding = decltype(encoding_constant)::value;
			auto table = std::make_unique<L1ContributionTable<kOut, encoding>>();
			table->build(weights.data(), kDimensions);
			for (auto rule : {DeclarationRule::Csa27, DeclarationRule::Csa24})
			{
				bool ok = true;
				for (const auto& [name, sfen] : positions)
					for (const char* side : {" b ", " w "})
					{
						std::string s = sfen;
						const auto  at = s.find(" b ") != std::string::npos ? s.find(" b ")
						                                                    : s.find(" w ");
						s.replace(at, 3, side);
						if (pos.set(s, &si).has_value())
							continue;
						std::uint8_t scalars[kDimensions];
						compute(pos, rule, encoding, scalars);
						const Color   us = pos.side_to_move();
						std::int32_t  by_table[kOut] = {};
						table->add(l1_inputs(king_declaration_state(pos, us, rule), us, encoding),
						           l1_inputs(king_declaration_state(pos, ~us, rule), ~us, encoding),
						           by_table);
						L1Inputs both[2];
						l1_inputs_both(pos, rule, encoding, both);
						const L1Inputs each[2] = {
						  l1_inputs(king_declaration_state(pos, us, rule), us, encoding),
						  l1_inputs(king_declaration_state(pos, ~us, rule), ~us, encoding)};
						ok &= std::memcmp(both, each, sizeof(both)) == 0;
						for (int o = 0; o < kOut; ++o)
						{
							std::int32_t direct = 0;
							for (int k = 0; k < kDimensions; ++k)
								direct += std::int32_t(weights[o * kDimensions + k]) * scalars[k];
							ok &= direct == by_table[o];
						}
					}
				tester.test(std::string(label) + " " + canonical_name(encoding) + " "
				              + canonical_name(rule),
				            ok);
			}
		};
		using Thermometer = std::integral_constant<ScalarEncoding, ScalarEncoding::Thermometer>;
		using OneHot      = std::integral_constant<ScalarEncoding, ScalarEncoding::OneHot>;
		check_encoding(Thermometer{}, "random");
		check_encoding(OneHot{}, "random");
		std::fill(weights.begin(), weights.end(), std::int8_t(-128));
		check_encoding(Thermometer{}, "all -128");
		check_encoding(OneHot{}, "all -128");
	}
}

}  // namespace Eval::NNUE::Features::NyugyokuScalar
}  // namespace YaneuraOu

#endif  // defined(EVAL_NNUE)
