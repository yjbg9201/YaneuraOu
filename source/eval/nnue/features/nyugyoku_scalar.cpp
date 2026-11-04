// 入玉スカラー特徴の計算と UnitTest

#include "nyugyoku_scalar.h"

#if defined(EVAL_NNUE)

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <vector>

#include "../../../misc.h"
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
// tatara 側と同じ整数式で計算し、浮動小数の丸め差を避ける。
std::uint8_t scale_to_u8(int num, int den) {
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
	const int          need       = s.required_points;
	const std::uint8_t hand_ratio = scale_to_u8(std::min(s.hand_points, need), need);

	if (encoding == ScalarEncoding::Thermometer)
	{
		for (int k = 1; k <= 8; ++k)
			out[k - 1] = flag(s.advance >= k);
		out[8] = flag(s.king_in_enemy_camp);
		for (int k = 1; k <= 10; ++k)
			out[8 + k] = flag(s.pieces_in_camp >= k);
		const int remaining           = s.remaining_points();
		const int remaining_potential = s.remaining_potential_points();
		for (int i = 0; i < kNumRemainingPointThresholds; ++i)
		{
			out[19 + i] = flag(remaining <= kRemainingPointThresholds[i]);
			out[29 + i] = flag(remaining_potential <= kRemainingPointThresholds[i]);
		}
		out[39] = hand_ratio;
		out[40] = flag(s.declarable());
		out[41] = flag(owner == BLACK);
	}
	else
	{
		out[s.advance] = kOne;
		out[9]         = flag(s.king_in_enemy_camp);
		out[10]        = scale_to_u8(std::min(s.pieces_in_camp, 10), 10);
		out[11]        = scale_to_u8(std::min(s.declaration_points, need), need);
		out[12]        = scale_to_u8(s.remaining_points(), need);
		out[13]        = hand_ratio;
		out[14]        = flag(s.declarable());
		out[15]        = flag(owner == BLACK);
	}
}

}  // namespace

KingDeclarationState king_declaration_state(const Position& pos, Color owner,
                                            DeclarationRule rule) {
	const Square   ksq  = pos.square<KING>(owner);
	const Bitboard camp = enemy_field(owner);
	const bool     king_in_camp = bool(camp & ksq);

	// 玉を含めて数え、玉の分を引く。大駒 (角・馬・飛・竜) は枚数×4 を足して 5 点にする。
	const int board_pieces   = pos.pieces(owner).pop_count() - 1;
	const int board_big      = pos.pieces(owner, BISHOP_HORSE, ROOK_DRAGON).pop_count();
	const int camp_pieces    = (pos.pieces(owner) & camp).pop_count() - (king_in_camp ? 1 : 0);
	const int camp_big       = (pos.pieces(owner, BISHOP_HORSE, ROOK_DRAGON) & camp).pop_count();
	const int hand           = hand_points(pos.hand_of(owner));
	const int rank           = int(rank_of(ksq));

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
// ファイルを更新したら両リポのコピーとこの値をそろえて更新する。
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
}

}  // namespace Eval::NNUE::Features::NyugyokuScalar
}  // namespace YaneuraOu

#endif  // defined(EVAL_NNUE)
