"""End-to-end engine tests against the real Stockfish binary.

Skipped when Stockfish is not installed, so the suite still runs on a bare
checkout.
"""

import asyncio
import shutil
import sys
import unittest
from pathlib import Path

import chess

MUAPPS = Path(__file__).resolve().parents[1] / "muapps"
if str(MUAPPS) not in sys.path:
    sys.path.insert(0, str(MUAPPS))

from chess_app.domain import CoachPolicy, build_ask, should_warn, solves_puzzle
from chess_app.engine import ChessEngine, EngineSettings, resolve_engine_path
from chess_app.events import ChessAsk

STOCKFISH = shutil.which("stockfish")

# 1.e4 e5 2.Nf3 Nc6 3.Bc4 Nd4 — white can win a piece with Nxd4.
AFTER_ND4 = "r1bqkbnr/pppp1ppp/8/4p3/2BnP3/5N2/PPPP1PPP/RNBQK2R w KQkq - 5 4"
# White to play and mate in one with Ra1-a8.
MATE_IN_ONE = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"
# 1.e4 e5 2.Qh5 g6 — white's queen on h5 hangs to gxh5 after any quiet move (Nc3).
QUEEN_HANGS = "rnbqkbnr/pppp1p1p/6p1/4p2Q/4P3/8/PPPP1PPP/RNB1KBNR w KQkq - 0 3"


class EnginePathTests(unittest.TestCase):
    def test_missing_binary_raises_before_any_session(self):
        with self.assertRaises(FileNotFoundError):
            resolve_engine_path("definitely-not-a-chess-engine")

    @unittest.skipIf(STOCKFISH is None, "stockfish not installed")
    def test_present_binary_resolves_to_an_absolute_path(self):
        self.assertTrue(Path(resolve_engine_path("stockfish")).is_absolute())


@unittest.skipIf(STOCKFISH is None, "stockfish not installed")
class ChessEngineTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = ChessEngine(
            EngineSettings(path="stockfish", skill_level=2, move_time_s=0.05, analyse_time_s=0.1)
        )
        self.policy = CoachPolicy()

    async def asyncTearDown(self):
        await self.engine.close()

    async def test_best_move_in_a_winning_position(self):
        # At a 0.1s budget the exact first choice varies, so assert what matters:
        # a legal recommendation, a white advantage, and Nxd4 among the top lines.
        ask = build_ask("best", chess.Board(AFTER_ND4), self.policy)
        verdict = await self.engine.answer(ask)
        board = chess.Board(AFTER_ND4)
        self.assertIn(chess.Move.from_uci(verdict.best_action), board.legal_moves)
        self.assertGreater(verdict.score, 0)
        self.assertEqual(verdict.delta, 0)
        self.assertEqual(len(verdict.detail["lines"]), 3)
        self.assertIn("Nxd4", [line["san"] for line in verdict.detail["lines"]])

    async def test_mate_in_one_is_reported_as_mate(self):
        ask = build_ask("best", chess.Board(MATE_IN_ONE), self.policy)
        verdict = await self.engine.answer(ask)
        self.assertEqual(verdict.best_action, "a1a8")
        self.assertEqual(verdict.detail["mate"], 1)

    async def test_good_candidate_move_has_no_delta(self):
        ask = build_ask("check", chess.Board(AFTER_ND4), self.policy, uci="f3d4", origin="guard")
        verdict = await self.engine.answer(ask)
        self.assertEqual(verdict.detail["san"], "Nxd4")
        self.assertLess(verdict.delta, 100)
        self.assertEqual(verdict.severity, "fine")

    async def test_blunder_candidate_move_is_scored_as_a_blunder(self):
        # White develops a knight and lets black take the h5 queen with gxh5.
        ask = build_ask("check", chess.Board(QUEEN_HANGS), self.policy, uci="b1c3", origin="guard")
        verdict = await self.engine.answer(ask)
        self.assertGreaterEqual(verdict.delta, 200)
        self.assertEqual(verdict.severity, "blunder")
        self.assertEqual(verdict.detail["reply_san"], "gxh5")

    async def test_guard_fires_on_that_blunder_and_stays_quiet_on_the_good_move(self):
        from chess_app.domain import ChessState

        state = ChessState()
        state.pending_uci = "b1c3"
        bad = await self.engine.answer(
            build_ask("check", chess.Board(QUEEN_HANGS), self.policy, uci="b1c3", origin="guard")
        )
        self.assertTrue(should_warn(bad, self.policy, state))

        state.pending_uci = "f3d4"
        good = await self.engine.answer(
            build_ask("check", chess.Board(AFTER_ND4), self.policy, uci="f3d4", origin="guard")
        )
        self.assertFalse(should_warn(good, self.policy, state))

    async def test_whatif_does_not_change_the_position(self):
        board = chess.Board(AFTER_ND4)
        ask = build_ask("whatif", board, self.policy, uci="f3e5")
        await self.engine.answer(ask)
        self.assertEqual(board.fen(), AFTER_ND4)

    async def test_move_that_delivers_mate_scores_as_a_win(self):
        ask = build_ask("check", chess.Board(MATE_IN_ONE), self.policy, uci="a1a8", origin="drill")
        verdict = await self.engine.answer(ask)
        self.assertEqual(verdict.delta, 0)
        self.assertTrue(solves_puzzle(MATE_IN_ONE, "a1a8", "mate", verdict))

    async def test_wrong_puzzle_answer_is_rejected(self):
        ask = build_ask("check", chess.Board(MATE_IN_ONE), self.policy, uci="g1h1", origin="drill")
        verdict = await self.engine.answer(ask)
        self.assertFalse(solves_puzzle(MATE_IN_ONE, "g1h1", "mate", verdict))

    async def test_asking_about_the_other_side_gets_an_answer(self):
        board = chess.Board()
        board.push_uci("e2e4")
        ask = build_ask("best", board, self.policy, side="white")
        self.assertIsInstance(ask, ChessAsk)
        verdict = await self.engine.answer(ask)
        self.assertTrue(verdict.best_action)
        self.assertEqual(chess.Board(ask.snapshot).turn, chess.WHITE)

    async def test_illegal_candidate_move_raises(self):
        ask = ChessAsk(kind="whatif", snapshot=AFTER_ND4, action="a1a8")
        with self.assertRaises(ValueError):
            await self.engine.answer(ask)

    async def test_engine_picks_a_legal_move_and_is_weak(self):
        uci = await self.engine.pick_move(AFTER_ND4)
        self.assertIn(chess.Move.from_uci(uci), chess.Board(AFTER_ND4).legal_moves)

    async def test_pick_move_returns_nothing_when_the_game_is_over(self):
        self.assertEqual(
            await self.engine.pick_move("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"),
            "",
        )

    async def test_start_is_idempotent_and_close_is_safe_twice(self):
        await self.engine.start()
        await self.engine.start()
        self.assertTrue(await self.engine.pick_move(AFTER_ND4))
        await self.engine.close()
        await self.engine.close()

    async def test_killed_engine_raises_instead_of_hanging(self):
        # Killing the process makes python-chess abandon an AnalysisResult whose
        # internal future holds the termination error. Nothing public reaches that
        # object, so roughly half of the runs print a stray "exception was never
        # retrieved" traceback here. The test still passes; the noise is the
        # library's, and it only happens because this test kills Stockfish itself.
        await self.engine.start()
        self.engine._transport.kill()
        with self.assertRaises(chess.engine.EngineError):
            await self.engine.answer(build_ask("best", chess.Board(AFTER_ND4), self.policy))

    async def test_concurrent_questions_are_serialised(self):
        asks = [
            build_ask("best", chess.Board(AFTER_ND4), self.policy),
            build_ask("best", chess.Board(MATE_IN_ONE), self.policy),
            build_ask("whatif", chess.Board(AFTER_ND4), self.policy, uci="f3e5"),
        ]
        verdicts = await asyncio.gather(*(self.engine.answer(ask) for ask in asks))
        self.assertIn(
            chess.Move.from_uci(verdicts[0].best_action),
            chess.Board(AFTER_ND4).legal_moves,
        )
        self.assertEqual(verdicts[1].best_action, "a1a8")
        self.assertEqual(verdicts[2].detail["san"], "Nxe5")


if __name__ == "__main__":
    unittest.main()
