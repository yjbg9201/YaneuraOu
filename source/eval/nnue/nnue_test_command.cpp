// NNUE評価関数に関するUSI拡張コマンド

#include "../../config.h"

#if defined(ENABLE_TEST_CMD) && defined(EVAL_NNUE)

#include "../../extra/all.h"
#include "evaluate_nnue.h"
#include "nnue_test_command.h"

#include <set>
#include <chrono>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "features/nyugyoku_scalar.h"

namespace YaneuraOu {
namespace Eval::NNUE {

namespace {

// Check the actual int16 accumulator, not only the rounded final score.
// Exercise both perspectives, promotions, captures, fallback paths and undo.
void TestAccumulator(Position&, std::istream& stream) {
  int games = 100;
  stream >> games;
  if (games <= 0) {
    std::cout << "test_accumulator: games must be positive" << std::endl;
    return;
  }
  constexpr int max_ply = 256;
  Position pos;
  auto states = std::make_unique<StateInfo[]>(max_ply);
  StateInfo root;
  PRNG prng(20261025);
  std::uint64_t moves = 0, pawn_moves = 0, pawn_captures = 0, promotions = 0, king_moves = 0, checksum = 0;
  for (int game = 0; game < games; ++game) {
    pos.set_hirate(&root);
    Eval::compute_eval(pos);
    for (int ply = 0; ply < max_ply; ++ply) {
      MoveList<LEGAL_ALL> legal(pos);
      if (legal.size() == 0) break;
      const Move move = legal.begin()[prng.rand(legal.size())];
      const auto parent = pos.state()->accumulator;
      pos.do_move(move, states[ply]);
      const auto& dirty = pos.state()->dirtyPiece;
      if (dirty.pieceNo[0] >= PIECE_NUMBER_KING) ++king_moves;
      for (int n = 0; n < dirty.dirty_num; ++n) {
        const int before = dirty.changed_piece[n].old_piece.fb;
        const int after = dirty.changed_piece[n].new_piece.fb;
        if (before >= f_pawn && before < f_pawn + 162) {
          if (after >= f_hand_pawn && after < f_hand_lance) ++pawn_captures;
          else {
            ++pawn_moves;
            if (after < f_pawn || after >= f_pawn + 162) ++promotions;
          }
        }
      }
      Eval::evaluate_with_no_return(pos);
#if defined(USE_AVX512) && defined(NNUE_HAS_SFNN16_ACCUMULATOR_PROPAGATE)
      // Check raw fc_0 outputs, including values hidden by FV_SCALE rounding.
      alignas(kCacheLineSize) TransformedFeatureType features[FeatureTransformer::kBufferSize];
      networks().feature_transformer.Transform(pos, features, false);
      const auto& fc = networks().network[moves % kLayerStacks].fc_0;
      using Fc = std::remove_cv_t<std::remove_reference_t<decltype(fc)>>;
      alignas(kCacheLineSize) typename Fc::OutputBuffer separate, fused;
      fc.Propagate(features, separate);
      fc.template PropagateSfnn16FromAccumulator<kTransformedFeatureDimensions>(
          pos.state()->accumulator.accumulation, pos.side_to_move(), fused);
      if (std::memcmp(separate, fused, sizeof(typename Fc::OutputType) * Fc::kOutputDimensions) != 0) {
        std::cout << "test_accumulator FAILED fused fc_0 game " << game << " ply " << ply << std::endl;
        return;
      }
#endif
      const auto incremental = pos.state()->accumulator;
      const Value score = Eval::evaluate(pos);
      const Value cached_reference = Eval::compute_eval(pos);
      if (score != cached_reference || std::memcmp(incremental.accumulation,
          pos.state()->accumulator.accumulation, sizeof(incremental.accumulation)) != 0) {
        std::cout << "test_accumulator FAILED cached refresh game " << game << " ply " << ply << std::endl;
        return;
      }
      networks().feature_transformer.RefreshAccumulatorForTest(pos);
      const Value reference = Eval::evaluate(pos);
      if (score != reference || std::memcmp(incremental.accumulation,
          pos.state()->accumulator.accumulation, sizeof(incremental.accumulation)) != 0) {
        std::cout << "test_accumulator FAILED game " << game << " ply " << ply
                  << " sfen " << pos.sfen() << std::endl;
        return;
      }
      checksum = checksum * 6364136223846793005ULL + static_cast<std::uint64_t>(score);
      pos.undo_move(move);
      if (std::memcmp(parent.accumulation, pos.state()->accumulator.accumulation,
                      sizeof(parent.accumulation)) != 0) {
        std::cout << "test_accumulator FAILED undo" << std::endl;
        return;
      }
      pos.do_move(move, states[ply]);
      Eval::evaluate_with_no_return(pos);
      if (std::memcmp(incremental.accumulation, pos.state()->accumulator.accumulation,
                      sizeof(incremental.accumulation)) != 0) {
        std::cout << "test_accumulator FAILED redo" << std::endl;
        return;
      }
      ++moves;
    }
  }
  pos.set_hirate(&root);
  std::cout << "test_accumulator passed: " << games << " games, " << moves
            << " moves, pawn moves " << pawn_moves << ", pawn captures " << pawn_captures
            << ", promotions " << promotions << ", king moves " << king_moves
            << ", checksum " << checksum << std::endl;
}

// 主に差分計算に関するRawFeaturesのテスト
void TestFeatures(Position&) {
  Position pos;
  const std::uint64_t num_games = 1000;
  StateInfo si;
  pos.set_hirate(&si);
  const int MAX_PLY = 256; // 256手までテスト

  StateInfo state[MAX_PLY]; // StateInfoを最大手数分だけ
  int ply; // 初期局面からの手数

  PRNG prng(20171128);

  std::uint64_t num_moves = 0;
  std::vector<std::uint64_t> num_updates(kRefreshTriggers.size() + 1);
  std::vector<std::uint64_t> num_resets(kRefreshTriggers.size());
  constexpr IndexType kUnknown = -1;
  std::vector<IndexType> trigger_map(RawFeatures::kDimensions, kUnknown);
  auto make_index_sets = [&](const Position& pos) {
    std::vector<std::vector<std::set<IndexType>>> index_sets(
        kRefreshTriggers.size(), std::vector<std::set<IndexType>>(2));
    for (IndexType i = 0; i < kRefreshTriggers.size(); ++i) {
      Features::IndexList active_indices[2];
      RawFeatures::AppendActiveIndices(pos, kRefreshTriggers[i],
                                       active_indices);
      for (const auto perspective : COLOR) {
        for (const auto index : active_indices[perspective]) {
          ASSERT(index < RawFeatures::kDimensions);
          ASSERT(index_sets[i][perspective].count(index) == 0);
          ASSERT(trigger_map[index] == kUnknown || trigger_map[index] == i);
          index_sets[i][perspective].insert(index);
          trigger_map[index] = i;
        }
      }
    }
    return index_sets;
  };
  auto update_index_sets = [&](const Position& pos, auto* index_sets) {
    for (IndexType i = 0; i < kRefreshTriggers.size(); ++i) {
      Features::IndexList removed_indices[2], added_indices[2];
      bool reset[2];
      RawFeatures::AppendChangedIndices(pos, kRefreshTriggers[i],
                                        removed_indices, added_indices, reset);
      for (const auto perspective : COLOR) {
        if (reset[perspective]) {
          (*index_sets)[i][perspective].clear();
          ++num_resets[i];
        } else {
          for (const auto index : removed_indices[perspective]) {
            ASSERT(index < RawFeatures::kDimensions);
            ASSERT((*index_sets)[i][perspective].count(index) == 1);
            ASSERT(trigger_map[index] == kUnknown || trigger_map[index] == i);
            (*index_sets)[i][perspective].erase(index);
            ++num_updates.back();
            ++num_updates[i];
            trigger_map[index] = i;
          }
        }
        for (const auto index : added_indices[perspective]) {
          ASSERT(index < RawFeatures::kDimensions);
          ASSERT((*index_sets)[i][perspective].count(index) == 0);
          ASSERT(trigger_map[index] == kUnknown || trigger_map[index] == i);
          (*index_sets)[i][perspective].insert(index);
          ++num_updates.back();
          ++num_updates[i];
          trigger_map[index] = i;
        }
      }
    }
  };

  std::cout << "feature set: " << RawFeatures::GetName()
            << "[" << RawFeatures::kDimensions << "]" << std::endl;
  std::cout << "start testing with random games";

  for (std::uint64_t i = 0; i < num_games; ++i) {
    auto index_sets = make_index_sets(pos);
    for (ply = 0; ply < MAX_PLY; ++ply) {
      MoveList<LEGAL_ALL> mg(pos); // 全合法手の生成

      // 合法な指し手がなかった == 詰み
      if (mg.size() == 0)
        break;

      // 生成された指し手のなかからランダムに選び、その指し手で局面を進める。
      Move m = mg.begin()[prng.rand(mg.size())];
      pos.do_move(m, state[ply]);

      ++num_moves;
      update_index_sets(pos, &index_sets);
      ASSERT(index_sets == make_index_sets(pos));
    }

    pos.set_hirate(&si);

    // 100回に1回ごとに'.'を出力(進んでいることがわかるように)
    if ((i % 100) == 0)
      std::cout << "." << std::flush;
  }
  std::cout << "passed." << std::endl;
  std::cout << num_games << " games, " << num_moves << " moves, "
            << num_updates.back() << " updates, "
            << (1.0 * num_updates.back() / num_moves)
            << " updates per move" << std::endl;
  std::size_t num_observed_indices = 0;
  for (IndexType i = 0; i < kRefreshTriggers.size(); ++i) {
    const auto count = std::count(trigger_map.begin(), trigger_map.end(), i);
    num_observed_indices += count;
    std::cout << "TriggerEvent(" << static_cast<int>(kRefreshTriggers[i])
              << "): " << count << " features ("
              << (100.0 * count / RawFeatures::kDimensions) << "%), "
              << num_updates[i] << " updates ("
              << (1.0 * num_updates[i] / num_moves) << " per move), "
              << num_resets[i] << " resets ("
              << (100.0 * num_resets[i] / num_moves) << "%)"
              << std::endl;
  }
  std::cout << "observed " << num_observed_indices << " ("
            << (100.0 * num_observed_indices / RawFeatures::kDimensions)
            << "% of " << RawFeatures::kDimensions
            << ") features" << std::endl;
}

// 評価関数の構造を表す文字列を出力する
void PrintInfo(std::istream& stream) {
  std::cout << "network architecture: " << GetArchitectureString() << std::endl;

  while (true) {
    std::string file_name;
    stream >> file_name;
    if (file_name.empty()) break;

    std::uint32_t hash_value;
    std::string architecture;
    const Tools::Result result = [&]() {
      std::ifstream file_stream(file_name, std::ios::binary);
      if (!file_stream) return Tools::Result(Tools::ResultCode::FileReadError);
	  return ReadHeader(file_stream, &hash_value, &architecture);
    }();

    std::cout << file_name << ": ";
    if (result.is_ok()) {
      if (hash_value == kHashValue) {
        std::cout << "matches with this binary";
        if (architecture != GetArchitectureString()) {
          std::cout << ", but architecture string differs: " << architecture;
        }
        std::cout << std::endl;
      } else {
        std::cout << architecture << std::endl;
      }
    } else {
      std::cout << "failed to read header" << std::endl;
    }
  }
}

}  // namespace

// NNUE評価関数に関するUSI拡張コマンド
// SFEN を 1 行 1 局面で読む ("sfen " の前置きは任意)。
std::vector<std::string> ReadSfens(const std::string& path) {
  std::vector<std::string> sfens;
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.rfind("sfen ", 0) == 0) line = line.substr(5);
    if (!line.empty()) sfens.push_back(line);
  }
  return sfens;
}

// 各局面の出力層の生の値と評価値を "生の値<TAB>評価値" で 1 行ずつ書く。
// 学習器の量子化 CPU forward との突合に使う。評価値は FV_SCALE オプションで割った値。
void EvalSfens(std::istream& stream) {
#if defined(SFNNwoPSQT)
  (void)stream;
  std::cout << "eval_sfens: not supported for this architecture" << std::endl;
#else
  std::string in_path, out_path;
  stream >> in_path >> out_path;
  const auto sfens = ReadSfens(in_path);
  std::ofstream out(out_path);
  if (sfens.empty() || !out) {
    std::cout << "eval_sfens: usage: test nn eval_sfens <in.sfen> <out.tsv>" << std::endl;
    return;
  }
  Position pos;
  StateInfo si;
  for (const auto& sfen : sfens) {
    if (pos.set(sfen, &si).has_value()) {
      std::cout << "eval_sfens: invalid sfen: " << sfen << std::endl;
      return;
    }
    const std::int64_t raw = ComputeRawOutputForTest(pos);
    const Value value = Eval::compute_eval(pos);
    out << raw << '\t' << int(value) << '\n';
  }
  std::cout << "eval_sfens: wrote " << sfens.size() << " positions to " << out_path << std::endl;
#endif
}

// 平手からのランダムプレイアウトで局面を集め、SFEN で書く。入玉に近い局面を含めるため、
// king_walk_percent % の対局では手番側の玉を前に進める手があれば優先して指す。
void GenSfens(std::istream& stream) {
  std::uint64_t count = 0, seed = 1;
  int king_walk_percent = 30;
  std::string out_path;
  stream >> count >> seed >> out_path >> king_walk_percent;
  std::ofstream out(out_path);
  if (count == 0 || seed == 0 || !out) {
    std::cout << "gen_sfens: usage: test nn gen_sfens <count> <seed (non-zero)> <out.sfen> "
                 "[king_walk_percent=30]" << std::endl;
    return;
  }
  constexpr int max_ply = 320;
  PRNG prng(seed);
  Position pos;
  StateInfo root;
  auto states = std::make_unique<StateInfo[]>(max_ply);
  std::uint64_t written = 0, entered = 0;
  while (written < count) {
    pos.set_hirate(&root);
    const bool king_walk = int(prng.rand(100)) < king_walk_percent;
    for (int ply = 0; ply < max_ply && written < count; ++ply) {
      MoveList<LEGAL_ALL> legal(pos);
      if (legal.size() == 0) break;
      Move move = legal.begin()[prng.rand(legal.size())];
      if (king_walk && prng.rand(100) < 70) {
        const Color us = pos.side_to_move();
        const Square ksq = pos.square<KING>(us);
        for (const auto& m : legal) {
          const Move mv = m;
          if (mv.from_sq() == ksq && !mv.is_drop()
              && relative_rank(us, rank_of(mv.to_sq())) < relative_rank(us, rank_of(ksq))) {
            move = mv;
            break;
          }
        }
      }
      pos.do_move(move, states[ply]);
      // 序盤の局面ばかりにならないよう、各手の後の局面を 1/4 の確率で採る。
      if (prng.rand(4) == 0) {
        out << pos.sfen() << '\n';
        ++written;
        for (const Color c : COLOR)
          if (enemy_field(c) & pos.square<KING>(c)) { ++entered; break; }
      }
    }
  }
  std::cout << "gen_sfens: wrote " << written << " positions (" << entered
            << " with a king in the enemy camp) to " << out_path << std::endl;
}

// 入玉スカラーの計算 1 回 (両玉分 96 次元) の時間を測る。
void BenchScalar(std::istream& stream) {
  std::string in_path;
  int repeat = 100;
  stream >> in_path >> repeat;
  const auto sfens = ReadSfens(in_path);
  if (sfens.empty() || repeat <= 0) {
    std::cout << "bench_scalar: usage: test nn bench_scalar <in.sfen> [repeat=100]" << std::endl;
    return;
  }
  std::vector<Position> positions(sfens.size());
  std::vector<StateInfo> states(sfens.size());
  for (size_t i = 0; i < sfens.size(); ++i) positions[i].set(sfens[i], &states[i]);
  namespace NS = Features::NyugyokuScalar;
  alignas(64) std::uint8_t out[NS::kDimensions];
  std::uint64_t checksum = 0;
  const auto start = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; ++r)
    for (const auto& p : positions) {
      NS::compute(p, NS::DeclarationRule::Csa27, NS::ScalarEncoding::Thermometer, out);
      checksum += out[r % NS::kDimensions];
    }
  const double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count();
  std::cout << "bench_scalar: " << ns / (double(repeat) * positions.size()) << " ns/call ("
            << positions.size() << " positions x " << repeat << ", checksum " << checksum << ")"
            << std::endl;
}

// 差分更新つきの評価 1 回の時間を測る。各局面の合法手ごとに do_move → evaluate → undo_move
// を繰り返す (親局面の accumulator は計算済みなので、探索中と同じく差分更新になる)。
void BenchEval(std::istream& stream) {
  std::string in_path;
  int repeat = 20;
  stream >> in_path >> repeat;
  const auto sfens = ReadSfens(in_path);
  if (sfens.empty() || repeat <= 0) {
    std::cout << "bench_eval: usage: test nn bench_eval <in.sfen> [repeat=20]" << std::endl;
    return;
  }
  Position pos;
  StateInfo root, st;
  std::uint64_t evals = 0;
  std::int64_t checksum = 0;
  double ns = 0;
  for (const auto& sfen : sfens) {
    pos.set(sfen, &root);
    MoveList<LEGAL_ALL> legal(pos);
    const auto start = std::chrono::steady_clock::now();
    for (int r = 0; r < repeat; ++r) {
      Eval::compute_eval(pos);
      for (const auto& m : legal) {
        pos.do_move(m, st);
        checksum += Eval::evaluate(pos);
        pos.undo_move(m);
      }
    }
    ns += std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count();
    evals += std::uint64_t(repeat) * legal.size();
  }
  std::cout << "bench_eval: " << ns / evals << " ns/eval (" << evals << " evaluations, checksum "
            << checksum << ")" << std::endl;
}

void TestCommand(Position& pos, std::istream& stream) {
  std::string sub_command;
  stream >> sub_command;

  if (sub_command == "test_features") {
    TestFeatures(pos);
  } else if (sub_command == "test_accumulator") {
    TestAccumulator(pos, stream);
  } else if (sub_command == "info") {
    PrintInfo(stream);
  } else if (sub_command == "eval_sfens") {
    EvalSfens(stream);
  } else if (sub_command == "gen_sfens") {
    GenSfens(stream);
  } else if (sub_command == "bench_scalar") {
    BenchScalar(stream);
  } else if (sub_command == "bench_eval") {
    BenchEval(stream);
  } else {
    std::cout << "usage:" << std::endl;
    std::cout << " test nn test_features" << std::endl;
    std::cout << " test nn test_accumulator [games=100]" << std::endl;
    std::cout << " test nn info [path/to/" << kFileName << "...]" << std::endl;
    std::cout << " test nn eval_sfens <in.sfen> <out.tsv>" << std::endl;
    std::cout << " test nn gen_sfens <count> <seed> <out.sfen> [king_walk_percent=30]" << std::endl;
    std::cout << " test nn bench_scalar <in.sfen> [repeat=100]" << std::endl;
    std::cout << " test nn bench_eval <in.sfen> [repeat=20]" << std::endl;
  }
}

} // namespace Eval::NNUE
} // namespace YaneuraOu

#endif  // defined(ENABLE_TEST_CMD) && defined(EVAL_NNUE)
