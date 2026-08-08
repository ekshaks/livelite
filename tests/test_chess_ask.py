"""Question-layer tests for the chess bundle: no Stockfish, no LLM, no network."""

import sys
import unittest
from pathlib import Path

import chess

MUAPPS = Path(__file__).resolve().parents[1] / "muapps"
if str(MUAPPS) not in sys.path:
    sys.path.insert(0, str(MUAPPS))

from chess_app.domain import (
    ChessState,
    CoachPolicy,
    build_ask,
    color_of,
    legal_move_in,
    read_verdict,
    should_warn,
    solves_puzzle,
    spend_budget,
    validated_board,
)
from chess_app.events import ChessAsk

from server.core.qa import Refusal, Verdict

# 1.e4 e5 2.Nf3 Nc6 3.Bc4 Nd4 — black has just played a bad knight move.
AFTER_ND4 = "r1bqkbnr/pppp1ppp/8/4p3/2BnP3/5N2/PPPP1PPP/RNBQK2R w KQkq - 5 4"
# White to play and mate in one: Ra1-a8.
MATE_IN_ONE = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"
# 1.f3 e5 2.g4 Qh4# — white is mated, the game is over.
FOOLS_MATE = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
# 1.e4 f5 2.Qh5+ — black is in check but not mated, so the turn cannot be flipped.
BLACK_IN_CHECK = "rnbqkbnr/ppppp1pp/8/5p1Q/4P3/8/PPPP1PPP/RNB1KBNR b KQkq - 1 2"


def played_board(*sans: str) -> chess.Board:
    """A board built by playing moves, so it has the history ``why`` rewinds into."""
    board = chess.Board()
    for san in sans:
        board.push_san(san)
    return board


class BoardHelperTests(unittest.TestCase):
    def test_legal_move_in_accepts_and_rejects(self):
        board = chess.Board()
        self.assertEqual(legal_move_in(board, "e2e4"), chess.Move.from_uci("e2e4"))
        self.assertIsNone(legal_move_in(board, "e2e5"))
        self.assertIsNone(legal_move_in(board, "banana"))
        self.assertIsNone(legal_move_in(board, ""))

    def test_validated_board_rejects_impossible_position(self):
        self.assertTrue(validated_board(AFTER_ND4).is_valid())
        with self.assertRaises(ValueError):
            validated_board("8/8/8/8/8/8/8/8 w - - 0 1")  # no kings
        with self.assertRaises(ValueError):
            validated_board("not a fen")

    def test_color_of(self):
        self.assertEqual(color_of("white"), chess.WHITE)
        self.assertEqual(color_of("Black"), chess.BLACK)
        self.assertIsNone(color_of(""))
        self.assertIsNone(color_of("purple"))


class ChessStateTests(unittest.TestCase):
    def test_push_and_undo_returns_to_kid_turn(self):
        state = ChessState()
        state.push_uci("e2e4")
        state.push_uci("e7e5")
        self.assertTrue(state.kid_to_move())
        self.assertEqual(state.undo_pair(), 2)
        self.assertEqual(state.board.fen(), chess.Board().fen())

    def test_undo_on_empty_stack_is_a_noop(self):
        state = ChessState()
        self.assertEqual(state.undo_pair(), 0)

    def test_illegal_push_raises_and_leaves_board_alone(self):
        state = ChessState()
        with self.assertRaises(ValueError):
            state.push_uci("e2e5")
        self.assertEqual(state.board.fen(), chess.Board().fen())

    def test_push_clears_pending_and_warned(self):
        state = ChessState()
        state.pending_uci = "e2e4"
        state.warned_uci = "e2e4"
        state.push_uci("e2e4")
        self.assertEqual((state.pending_uci, state.warned_uci), ("", ""))

    def test_set_position_validates(self):
        state = ChessState()
        state.set_position(MATE_IN_ONE)
        self.assertEqual(state.board.fen(), MATE_IN_ONE)
        with self.assertRaises(ValueError):
            state.set_position("8/8/8/8/8/8/8/8 w - - 0 1")

    def test_kid_to_move_follows_kid_color(self):
        state = ChessState(kid_color=chess.BLACK)
        self.assertFalse(state.kid_to_move())
        state.push_uci("e2e4")
        self.assertTrue(state.kid_to_move())


class BuildAskTests(unittest.TestCase):
    def setUp(self):
        self.policy = CoachPolicy()
        self.state = ChessState()
        self.board = chess.Board(AFTER_ND4)

    def test_every_kind_builds(self):
        for kind in ("best", "hint", "explain"):
            ask = build_ask(kind, self.board, self.policy)
            self.assertIsInstance(ask, ChessAsk)
            self.assertEqual(ask.kind, kind)
            self.assertEqual(ask.snapshot, self.board.fen())
            self.assertTrue(ask.request_id)
        for kind in ("whatif", "check"):
            ask = build_ask(kind, self.board, self.policy, uci="f3d4")
            self.assertIsInstance(ask, ChessAsk)
            self.assertEqual(ask.action, "f3d4")

    def test_why_asks_about_the_position_before_the_move(self):
        board = played_board("e4", "e5", "Nf3", "Nc6", "Bc4", "Nd4")
        ask = build_ask("why", board, self.policy)
        self.assertIsInstance(ask, ChessAsk)
        # The move asked about is the one just played, and the snapshot is the
        # position it was played from — where it is still legal.
        self.assertEqual(ask.action, "c6d4")
        before = chess.Board(ask.snapshot)
        self.assertEqual(before.turn, chess.BLACK)
        self.assertIn(chess.Move.from_uci("c6d4"), before.legal_moves)

    def test_why_refused_before_any_move_is_played(self):
        refusal = build_ask("why", chess.Board(), self.policy)
        self.assertIsInstance(refusal, Refusal)
        self.assertEqual(refusal.reason, "no_last_move")

    def test_why_ignores_a_spoken_move_and_uses_the_history(self):
        board = played_board("e4", "e5", "Nf3")
        ask = build_ask("why", board, self.policy, uci="a1a8")
        self.assertIsInstance(ask, ChessAsk)
        self.assertEqual(ask.action, "g1f3")

    def test_unknown_kind_refused(self):
        refusal = build_ask("waffle", self.board, self.policy)
        self.assertIsInstance(refusal, Refusal)
        self.assertEqual(refusal.reason, "unknown_kind")

    def test_live_board_is_never_mutated(self):
        before = self.board.fen()
        build_ask("whatif", self.board, self.policy, uci="f3d4")
        self.assertEqual(self.board.fen(), before)

    def test_illegal_candidate_move_refused(self):
        refusal = build_ask("whatif", self.board, self.policy, uci="a1a8")
        self.assertIsInstance(refusal, Refusal)
        self.assertEqual(refusal.reason, "illegal_action")

    def test_move_given_for_a_kind_that_takes_none_is_refused(self):
        refusal = build_ask("best", self.board, self.policy, uci="f3d4")
        self.assertIsInstance(refusal, Refusal)
        self.assertEqual(refusal.reason, "unexpected_action")

    def test_ask_about_the_other_side_flips_the_copy(self):
        board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1")
        ask = build_ask("best", board, self.policy, side="white")
        self.assertIsInstance(ask, ChessAsk)
        self.assertEqual(ask.side(), "white")
        self.assertTrue(chess.Board(ask.snapshot).turn == chess.WHITE)

    def test_ask_about_the_side_to_move_keeps_the_position(self):
        ask = build_ask("best", self.board, self.policy, side="white")
        self.assertIsInstance(ask, ChessAsk)
        self.assertEqual(ask.snapshot, self.board.fen())

    def test_unknown_side_refused(self):
        refusal = build_ask("best", self.board, self.policy, side="purple")
        self.assertIsInstance(refusal, Refusal)
        self.assertEqual(refusal.reason, "unknown_side")

    def test_flip_refused_when_side_to_move_is_in_check(self):
        board = chess.Board(BLACK_IN_CHECK)
        self.assertTrue(board.is_check())
        self.assertFalse(board.is_checkmate())
        refusal = build_ask("best", board, self.policy, side="white")
        self.assertIsInstance(refusal, Refusal)
        self.assertEqual(refusal.reason, "cannot_flip_turn")

    def test_asking_about_the_checked_side_itself_is_fine(self):
        board = chess.Board(BLACK_IN_CHECK)
        ask = build_ask("best", board, self.policy, side="black")
        self.assertIsInstance(ask, ChessAsk)
        self.assertEqual(ask.snapshot, BLACK_IN_CHECK)

    def test_finished_game_refused(self):
        mated = chess.Board(FOOLS_MATE)
        self.assertTrue(mated.is_game_over())
        refusal = build_ask("best", mated, self.policy)
        self.assertIsInstance(refusal, Refusal)
        self.assertEqual(refusal.reason, "game_over")

    def test_puzzle_goal_is_carried(self):
        ask = build_ask(
            "check",
            chess.Board(MATE_IN_ONE),
            self.policy,
            uci="a1a8",
            origin="drill",
            goal="mate",
        )
        self.assertEqual(ask.extra["goal"], "mate")
        self.assertEqual(ask.origin, "drill")


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.board = chess.Board(AFTER_ND4)

    def test_best_downgrades_to_hint_when_budget_spent(self):
        policy = CoachPolicy(ask_best_budget=2)
        state = ChessState()
        kinds = [self.answer("best", policy, state) for _ in range(4)]
        self.assertEqual(kinds, ["best", "best", "hint", "hint"])
        self.assertEqual(state.best_asks_used, 2)

    def answer(self, kind: str, policy: CoachPolicy, state: ChessState) -> str:
        """Ask a question and charge it, the way the workflow does once it replies."""
        ask = build_ask(kind, self.board, policy, state)
        spend_budget(ask, state)
        return ask.kind

    def test_a_failed_question_costs_nothing(self):
        policy = CoachPolicy(ask_best_budget=1)
        state = ChessState()
        # Built but never answered — the engine failed, so the budget must not move.
        build_ask("best", self.board, policy, state)
        self.assertEqual(state.best_asks_used, 0)
        self.assertEqual(self.answer("best", policy, state), "best")

    def test_zero_budget_never_answers_best(self):
        policy = CoachPolicy(ask_best_budget=0)
        state = ChessState()
        self.assertEqual(build_ask("best", self.board, policy, state).kind, "hint")

    def test_negative_budget_is_unlimited(self):
        policy = CoachPolicy(ask_best_budget=-1)
        state = ChessState()
        kinds = [self.answer("best", policy, state) for _ in range(5)]
        self.assertEqual(set(kinds), {"best"})

    def test_hint_budget_downgrades_to_explain(self):
        policy = CoachPolicy(hint_budget=1)
        state = ChessState()
        kinds = [self.answer("hint", policy, state) for _ in range(3)]
        self.assertEqual(kinds, ["hint", "explain", "explain"])

    def test_guard_and_puzzle_asks_do_not_spend_budget(self):
        policy = CoachPolicy(ask_best_budget=0)
        state = ChessState()
        ask = build_ask("check", self.board, policy, state, uci="f3d4", origin="guard")
        self.assertEqual(ask.kind, "check")
        self.assertEqual(state.best_asks_used, 0)

    def test_no_state_means_no_budget_tracking(self):
        policy = CoachPolicy(ask_best_budget=0)
        self.assertEqual(build_ask("best", self.board, policy).kind, "best")


class VerdictTests(unittest.TestCase):
    def setUp(self):
        self.ask = ChessAsk(kind="check", snapshot=AFTER_ND4, action="f3d4")

    def test_delta_is_the_loss_against_best(self):
        verdict = read_verdict(self.ask, score_cp=120, best_uci="f3d4", action_score_cp=-300)
        self.assertEqual(verdict.delta, 420)
        self.assertEqual(verdict.severity, "blunder")
        self.assertEqual(verdict.score, -300)
        self.assertEqual(verdict.best_action, "f3d4")

    def test_a_better_than_best_move_never_goes_negative(self):
        verdict = read_verdict(self.ask, score_cp=100, best_uci="f3d4", action_score_cp=150)
        self.assertEqual(verdict.delta, 0)
        self.assertEqual(verdict.severity, "fine")

    def test_no_candidate_move_means_no_delta(self):
        verdict = read_verdict(ChessAsk(kind="best", snapshot=AFTER_ND4), 80, "f3d4")
        self.assertEqual((verdict.delta, verdict.score, verdict.severity), (0, 80, "fine"))

    def test_severity_bands(self):
        bands = {
            0: "fine",
            49: "fine",
            50: "inaccuracy",
            120: "mistake",
            250: "blunder",
        }
        for delta, expected in bands.items():
            verdict = read_verdict(self.ask, 0, "f3d4", action_score_cp=-delta)
            self.assertEqual(verdict.severity, expected, delta)

    def test_detail_is_copied_not_shared(self):
        detail = {"mate": 3}
        verdict = read_verdict(self.ask, 0, "a1a8", detail=detail)
        detail["mate"] = 99
        self.assertEqual(verdict.detail, {"mate": 3})


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.state = ChessState()
        self.state.pending_uci = "f3d4"

    def test_warns_at_or_above_threshold(self):
        policy = CoachPolicy(guard_threshold_cp=200)
        self.assertTrue(should_warn(Verdict(delta=200), policy, self.state))
        self.assertTrue(should_warn(Verdict(delta=900), policy, self.state))
        self.assertFalse(should_warn(Verdict(delta=199), policy, self.state))

    def test_disabled_guard_never_warns(self):
        policy = CoachPolicy(guard_enabled=False, guard_threshold_cp=0)
        self.assertFalse(should_warn(Verdict(delta=5000), policy, self.state))

    def test_threshold_is_configurable_down_to_always(self):
        policy = CoachPolicy(guard_threshold_cp=0)
        self.assertTrue(should_warn(Verdict(delta=0), policy, self.state))

    def test_warns_only_once_per_move(self):
        policy = CoachPolicy()
        verdict = Verdict(delta=500)
        self.assertTrue(should_warn(verdict, policy, self.state))
        self.state.warned_uci = self.state.pending_uci
        self.assertFalse(should_warn(verdict, policy, self.state))
        self.state.pending_uci = "c6d4"
        self.assertTrue(should_warn(verdict, policy, self.state))


class PuzzleJudgeTests(unittest.TestCase):
    def test_mate_goal_needs_actual_mate(self):
        self.assertTrue(solves_puzzle(MATE_IN_ONE, "a1a8", "mate", Verdict()))
        self.assertFalse(solves_puzzle(MATE_IN_ONE, "g1h1", "mate", Verdict()))

    def test_illegal_answer_never_solves(self):
        self.assertFalse(solves_puzzle(MATE_IN_ONE, "a1a9", "mate", Verdict()))
        self.assertFalse(solves_puzzle(MATE_IN_ONE, "h7h8", "mate", Verdict()))

    def test_material_goal_accepts_any_move_that_keeps_the_edge(self):
        self.assertTrue(solves_puzzle(AFTER_ND4, "f3d4", "win_material", Verdict(delta=0)))
        self.assertFalse(solves_puzzle(AFTER_ND4, "f3d4", "win_material", Verdict(delta=300)))


if __name__ == "__main__":
    unittest.main()
