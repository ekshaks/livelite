"""End-to-end tests for the chess workflow.

These drive the real :class:`~chess_app.game_async.ChessWorkflow` through a real
:class:`~server.core.async_controller_flow.AsyncControllerFlow` and a real
:class:`~server.core.effects.EffectRunner`, with a small deterministic analyser
standing in for Stockfish so a whole game runs in milliseconds and the assertions
are about behaviour, not timing.

The analyser is not a mock of the engine's interface: it is a second, simpler
implementation of "score this position" written with python-chess, used because the
tests are about the *workflow's* decisions. Engine behaviour itself is covered
against the real binary in ``test_chess_engine.py``.
"""

import asyncio
import sys
import unittest
from pathlib import Path
from typing import Self

import chess
import reactivex

MUAPPS = Path(__file__).resolve().parents[1] / "muapps"
if str(MUAPPS) not in sys.path:
    sys.path.insert(0, str(MUAPPS))

from chess_app.coach import prompt_for
from chess_app.domain import CoachPolicy, read_verdict
from chess_app.events import (
    AnalysisFailed,
    AnalysisReady,
    CoachLineReady,
    ConfirmMove,
    EngineMoveReady,
    MoveDragged,
    MoveSpoken,
    NewGame,
    NextPuzzle,
    PositionObserved,
    QuestionAsked,
    RepeatLast,
    RequestAnalysis,
    RequestCoachLine,
    RequestEngineMove,
    StopSession,
    TakeBackMove,
    UnclearInput,
    Undo,
)
from chess_app.game_async import ChessWorkflow
from chess_app.puzzles import Puzzle, PuzzleBook, load_puzzles

from server.core.async_controller_flow import AsyncControllerFlow
from server.core.effects import EffectRunner

PUZZLES_FILE = Path(__file__).resolve().parents[1] / "muapps" / "chess_app" / "puzzles.yml"

#: Material values used by the stand-in analyser, in centipawns.
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 300,
    chess.BISHOP: 320,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


def material_score(board: chess.Board) -> int:
    """Score a position by material, from the side to move's point of view.

    Args:
        board: The position to score.

    Returns:
        Centipawns; positive means the side to move has more material.
    """
    if board.is_checkmate():
        return -10000
    total = 0
    for square, piece in board.piece_map().items():
        del square
        value = PIECE_VALUES[piece.piece_type]
        total += value if piece.color == board.turn else -value
    return total


def best_reply(board: chess.Board) -> tuple[int, str]:
    """Pick the move that maximises material after the opponent's best answer.

    Args:
        board: The position to search.

    Returns:
        ``(score, uci)`` for the chosen move.
    """
    best = (-100000, "")
    for move in board.legal_moves:
        board.push(move)
        if board.is_checkmate():
            score = 10000
        else:
            worst = min(_after(board, reply) for reply in board.legal_moves) if any(
                board.legal_moves
            ) else -material_score(board)
            score = worst
        board.pop()
        if score > best[0]:
            best = (score, move.uci())
    return best


def _after(board: chess.Board, move: chess.Move) -> int:
    """Material for the side that just moved, after the opponent's best answer.

    The opponent's reply has to be included, otherwise hanging a queen scores the
    same as a quiet developing move and nothing ever looks like a blunder.

    Args:
        board: The position before ``move``.
        move: The move to evaluate.

    Returns:
        Centipawns from the point of view of the side playing ``move``.
    """
    board.push(move)
    replies = list(board.legal_moves)
    if not replies:
        score = -material_score(board)
    else:
        score = -max(_material_after(board, reply) for reply in replies)
    board.pop()
    return score


def _material_after(board: chess.Board, move: chess.Move) -> int:
    """Material for the side playing ``move``, right after it is played."""
    board.push(move)
    score = -material_score(board)
    board.pop()
    return score


async def analyse(request: RequestAnalysis) -> AnalysisReady:
    """Answer one ask with the material analyser.

    Args:
        request: The analysis request.

    Returns:
        The reply event.
    """
    ask = request.ask
    board = chess.Board(ask.snapshot)
    score, best_uci = best_reply(board)
    detail = {"lines": [{"uci": best_uci, "san": board.san(chess.Move.from_uci(best_uci)),
                         "score": score}], "mate": None}
    action_score = None
    if ask.action:
        move = chess.Move.from_uci(ask.action)
        detail["san"] = board.san(move)
        action_score = _after(board, move)
    verdict = read_verdict(ask, score, best_uci, action_score, detail)
    return AnalysisReady(ask.request_id, verdict)


async def engine_move(request: RequestEngineMove) -> EngineMoveReady:
    """Pick a reply for the app's side with the material analyser."""
    board = chess.Board(request.fen)
    if board.is_game_over():
        return EngineMoveReady(request.request_id, "")
    return EngineMoveReady(request.request_id, best_reply(board)[1])


async def coach_line(request: RequestCoachLine) -> CoachLineReady:
    """Stand in for the coach model, but pick the prompt the way production does.

    Using the real :func:`~chess_app.coach.prompt_for` means these tests check the
    actual prompt-selection rules, not a label the workflow happened to pass.

    Args:
        request: The coaching request.

    Returns:
        A deterministic line naming the prompt that would have been used.
    """
    prompt_id = prompt_for(request.ask, request.facts.get("prompt_id", ""))
    return CoachLineReady(request.request_id, f"[{prompt_id}] {request.ask.kind}")


class Harness:
    """Drives a workflow and collects everything it said.

    Args:
        workflow: The controller under test.
    """

    def __init__(self, workflow: ChessWorkflow):
        self.workflow = workflow
        self.flow = AsyncControllerFlow(workflow, name="test_chess")
        self.outputs = []
        self.effects = EffectRunner(self.flow.submit, name="test_effects")
        self.effects.register(RequestAnalysis, analyse)
        self.effects.register(RequestEngineMove, engine_move)
        self.effects.register(RequestCoachLine, coach_line)

    async def __aenter__(self) -> Self:
        self.flow.outputs.observable.subscribe(on_next=self._record)
        self.flow.outputs.to(self.effects.sink(), name="test_effects")
        self.flow.input_sink()(reactivex.never())
        self.flow.start()
        await self.settle()
        return self

    async def __aexit__(self, *error) -> None:
        await self.effects.close()
        self.flow.close()

    def _record(self, item) -> None:
        self.outputs.append(item)

    async def send(self, *events) -> None:
        """Submit events and let the workflow finish reacting.

        Args:
            *events: Events to submit in order.
        """
        for event in events:
            self.flow.submit(event)
        await self.settle()

    async def settle(self, rounds: int = 40) -> None:
        """Let queued tasks and effects run to quiescence.

        Args:
            rounds: How many event-loop turns to yield.
        """
        for _ in range(rounds):
            await asyncio.sleep(0)

    @property
    def said(self) -> list[str]:
        """Every line the workflow spoke, in order."""
        return [message for item in self.outputs for message in item.messages]

    @property
    def last_fen(self) -> str:
        """The most recent position pushed to the browser."""
        for item in reversed(self.outputs):
            if item.feedback is not None and item.feedback.name == "chess_position":
                return item.feedback.data["fen"]
        return ""

    @property
    def statuses(self) -> list[str]:
        """Every status value pushed to the browser."""
        return [
            item.feedback.result
            for item in self.outputs
            if item.feedback is not None and item.feedback.name == "chess_status"
        ]

    def spoke(self, fragment: str) -> bool:
        """Whether any spoken line contains ``fragment``."""
        return any(fragment in line for line in self.said)


MATE_PUZZLE = Puzzle(
    "mate-1",
    "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1",
    "mate",
    "White mates in one.",
    "Look at the back row.",
)

# After 1.e4 e5 2.Nf3, black to move.  ``d8h4`` throws the queen away to
# ``Nxh4`` and is *not* checkmate; ``b8c6`` is the quiet control move.
QUEEN_HANGS = "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"


def puzzle_workflow(*puzzles: Puzzle, **policy_kwargs) -> ChessWorkflow:
    """Build a puzzle-mode workflow over the given puzzles."""
    return ChessWorkflow(
        policy=CoachPolicy(**policy_kwargs),
        book=PuzzleBook(puzzles or (MATE_PUZZLE,)),
        mode="puzzle",
    )


def play_workflow(fen: str = "", **policy_kwargs) -> ChessWorkflow:
    """Build a play-mode workflow, optionally from a given position."""
    workflow = ChessWorkflow(policy=CoachPolicy(**policy_kwargs), mode="play")
    if fen:
        workflow.state.set_position(fen)
        workflow.state.kid_color = workflow.state.board.turn
    return workflow


class PuzzleModeTest(unittest.IsolatedAsyncioTestCase):
    async def test_speaks_the_goal_and_pushes_the_position(self):
        async with Harness(puzzle_workflow()) as harness:
            self.assertTrue(harness.spoke("White mates in one."))
            self.assertEqual(harness.last_fen, MATE_PUZZLE.fen)
            self.assertIn("puzzle", harness.statuses)

    async def test_correct_answer_is_praised(self):
        async with Harness(puzzle_workflow()) as harness:
            await harness.send(MoveSpoken("a1a8"))
            self.assertTrue(harness.spoke("coach_puzzle_right"))
            self.assertFalse(harness.spoke("coach_puzzle_wrong"))

    async def test_first_wrong_answer_gets_a_nudge_not_the_answer(self):
        async with Harness(puzzle_workflow()) as harness:
            await harness.send(MoveSpoken("g1g2"))
            self.assertTrue(harness.spoke("coach_puzzle_wrong"))
            self.assertFalse(harness.spoke("The move was"))

    async def test_second_wrong_answer_shows_the_answer_with_an_arrow(self):
        async with Harness(puzzle_workflow()) as harness:
            await harness.send(MoveSpoken("g1g2"))
            await harness.send(MoveSpoken("g1h2"))
            self.assertTrue(harness.spoke("The move was"))
            arrows = [
                item.feedback.data["arrow"]
                for item in harness.outputs
                if item.feedback is not None and item.feedback.name == "chess_position"
            ]
            self.assertIn("a1a8", arrows)

    async def test_illegal_move_is_refused_and_the_puzzle_continues(self):
        async with Harness(puzzle_workflow()) as harness:
            await harness.send(MoveSpoken("e2e4"))
            self.assertTrue(harness.spoke("not possible"))
            await harness.send(MoveSpoken("a1a8"))
            self.assertTrue(harness.spoke("coach_puzzle_right"))

    async def test_unclear_speech_asks_again(self):
        async with Harness(puzzle_workflow()) as harness:
            await harness.send(UnclearInput("mmm"))
            self.assertTrue(harness.spoke("did not catch"))

    async def test_next_puzzle_advances(self):
        second = Puzzle("mate-2", "6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1", "mate", "Again.", "")
        async with Harness(puzzle_workflow(MATE_PUZZLE, second)) as harness:
            await harness.send(MoveSpoken("a1a8"))
            await harness.send(NextPuzzle())
            self.assertEqual(harness.last_fen, second.fen)
            self.assertTrue(harness.spoke("Again."))

    async def test_a_question_during_a_puzzle_is_answered_then_the_puzzle_resumes(self):
        async with Harness(puzzle_workflow()) as harness:
            await harness.send(QuestionAsked("hint"))
            self.assertTrue(harness.spoke("coach_hint"))
            await harness.send(MoveSpoken("a1a8"))
            self.assertTrue(harness.spoke("coach_puzzle_right"))

    async def test_every_shipped_puzzle_can_be_solved(self):
        book = load_puzzles(PUZZLES_FILE)
        for puzzle in book.puzzles:
            if puzzle.goal != "mate":
                continue
            board = chess.Board(puzzle.fen)
            mates = []
            for move in board.legal_moves:
                board.push(move)
                if board.is_checkmate():
                    mates.append(move.uci())
                board.pop()
            self.assertTrue(mates, f"{puzzle.id} has no mate in one")
            async with Harness(puzzle_workflow(puzzle)) as harness:
                await harness.send(MoveSpoken(mates[0]))
                self.assertTrue(harness.spoke("coach_puzzle_right"), puzzle.id)


class BlunderGuardTest(unittest.IsolatedAsyncioTestCase):
    async def test_a_quiet_move_is_played_without_a_warning(self):
        async with Harness(play_workflow(QUEEN_HANGS)) as harness:
            await harness.send(MoveSpoken("b8c6"))
            self.assertTrue(harness.spoke("You played"))
            self.assertFalse(harness.spoke("coach_guard"))

    async def test_losing_the_queen_warns_exactly_once(self):
        async with Harness(play_workflow(QUEEN_HANGS)) as harness:
            await harness.send(MoveSpoken("d8h4"))
            self.assertEqual(harness.said.count("[coach_guard] check"), 1)
            self.assertIn("warned", harness.statuses)
            self.assertFalse(harness.spoke("You played"))

    async def test_confirming_a_warned_move_plays_it(self):
        async with Harness(play_workflow(QUEEN_HANGS)) as harness:
            await harness.send(MoveSpoken("d8h4"))
            await harness.send(ConfirmMove())
            self.assertTrue(harness.spoke("You played"))
            queen = chess.Board(harness.last_fen).piece_at(chess.H4)
            self.assertEqual(queen, chess.Piece(chess.QUEEN, chess.BLACK))

    async def test_taking_back_a_warned_move_keeps_the_turn(self):
        async with Harness(play_workflow(QUEEN_HANGS)) as harness:
            await harness.send(MoveSpoken("d8h4"))
            await harness.send(TakeBackMove())
            self.assertTrue(harness.spoke("still your turn"))
            self.assertEqual(harness.last_fen, QUEEN_HANGS)

    async def test_the_guard_can_be_switched_off(self):
        async with Harness(play_workflow(QUEEN_HANGS, guard_enabled=False)) as harness:
            await harness.send(MoveSpoken("d8h4"))
            self.assertFalse(harness.spoke("coach_guard"))
            self.assertTrue(harness.spoke("You played"))

    async def test_a_high_threshold_lets_the_blunder_through(self):
        async with Harness(play_workflow(QUEEN_HANGS, guard_threshold_cp=5000)) as harness:
            await harness.send(MoveSpoken("d8h4"))
            self.assertFalse(harness.spoke("coach_guard"))
            self.assertTrue(harness.spoke("You played"))

    async def test_changing_your_mind_re_checks_the_new_move(self):
        async with Harness(play_workflow(QUEEN_HANGS)) as harness:
            await harness.send(MoveSpoken("d8h4"))
            await harness.send(MoveDragged("b8c6"))
            self.assertTrue(harness.spoke("You played"))
            self.assertEqual(harness.said.count("[coach_guard] check"), 1)


class PlayModeTest(unittest.IsolatedAsyncioTestCase):
    async def test_the_app_replies_with_its_own_move(self):
        async with Harness(play_workflow()) as harness:
            await harness.send(MoveSpoken("e2e4"))
            self.assertTrue(harness.spoke("I played"))
            self.assertIn("thinking", harness.statuses)

    async def test_undo_returns_the_turn_to_the_kid(self):
        async with Harness(play_workflow()) as harness:
            await harness.send(MoveSpoken("e2e4"))
            await harness.send(Undo())
            self.assertTrue(harness.spoke("Taken back."))
            self.assertTrue(harness.workflow.state.kid_to_move())

    async def test_undo_with_nothing_to_undo_says_so(self):
        async with Harness(play_workflow()) as harness:
            await harness.send(Undo())
            self.assertTrue(harness.spoke("nothing to take back"))

    async def test_new_game_resets_the_board_and_the_budgets(self):
        async with Harness(play_workflow()) as harness:
            harness.workflow.state.best_asks_used = 3
            await harness.send(MoveSpoken("e2e4"))
            await harness.send(NewGame())
            self.assertEqual(harness.last_fen, chess.Board().fen())
            self.assertEqual(harness.workflow.state.best_asks_used, 0)

    async def test_a_position_from_the_browser_is_adopted(self):
        async with Harness(play_workflow()) as harness:
            await harness.send(PositionObserved(QUEEN_HANGS))
            self.assertEqual(harness.last_fen, QUEEN_HANGS)
            self.assertTrue(harness.spoke("Got the position."))

    async def test_an_illegal_position_from_outside_is_rejected(self):
        async with Harness(play_workflow()) as harness:
            await harness.send(PositionObserved("not a fen at all"))
            self.assertTrue(harness.spoke("does not look legal"))

    async def test_repeat_says_the_last_line_again(self):
        async with Harness(play_workflow()) as harness:
            await harness.send(RepeatLast())
            self.assertGreaterEqual(harness.said.count("Your turn."), 2)

    async def test_stop_finishes_the_session(self):
        async with Harness(play_workflow()) as harness:
            await harness.send(StopSession())
            self.assertTrue(harness.spoke("Good game."))
            self.assertTrue(any(item.finished for item in harness.outputs))

    async def test_checkmate_by_the_kid_is_celebrated(self):
        # Black is one move from mating with Qh4.
        fen = "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2"
        async with Harness(play_workflow(fen, guard_enabled=False)) as harness:
            await harness.send(MoveSpoken("d8h4"))
            self.assertTrue(harness.spoke("You won"))
            self.assertIn("game_over", harness.statuses)


class QuestionTest(unittest.IsolatedAsyncioTestCase):
    async def test_best_move_answer_draws_an_arrow(self):
        async with Harness(play_workflow()) as harness:
            await harness.send(QuestionAsked("best"))
            self.assertTrue(harness.spoke("coach_best"))
            arrows = [
                item.feedback.data["arrow"]
                for item in harness.outputs
                if item.feedback is not None and item.feedback.name == "chess_position"
            ]
            self.assertTrue(any(arrow for arrow in arrows))

    async def test_hint_never_reveals_the_move(self):
        async with Harness(play_workflow()) as harness:
            await harness.send(QuestionAsked("hint"))
            self.assertTrue(harness.spoke("coach_hint"))

    async def test_what_if_does_not_play_the_move(self):
        async with Harness(play_workflow()) as harness:
            await harness.send(QuestionAsked("whatif", action_text="e2e4"))
            self.assertTrue(harness.spoke("coach_whatif"))
            self.assertEqual(harness.last_fen, chess.Board().fen())

    async def test_what_if_about_an_illegal_move_is_refused(self):
        async with Harness(play_workflow()) as harness:
            await harness.send(QuestionAsked("whatif", action_text="e2e5"))
            self.assertTrue(harness.spoke("isn't possible"))

    async def test_asking_about_the_other_side_is_answered(self):
        async with Harness(play_workflow()) as harness:
            await harness.send(MoveSpoken("e2e4"))
            await harness.send(QuestionAsked("best", side="black"))
            self.assertTrue(harness.spoke("coach_best"))

    async def test_the_best_move_budget_downgrades_to_a_hint(self):
        async with Harness(play_workflow(ask_best_budget=1)) as harness:
            await harness.send(QuestionAsked("best"))
            await harness.send(QuestionAsked("best"))
            self.assertTrue(harness.spoke("coach_best"))
            self.assertTrue(harness.spoke("coach_hint"))

    async def test_an_unlimited_budget_keeps_answering(self):
        async with Harness(play_workflow(ask_best_budget=-1)) as harness:
            await harness.send(QuestionAsked("best"), QuestionAsked("best"))
            self.assertEqual(harness.said.count("[coach_best] best"), 2)
            self.assertFalse(harness.spoke("coach_hint"))

    async def test_an_engine_failure_apologises_and_carries_on(self):
        workflow = play_workflow()
        harness = Harness(workflow)
        harness.effects = EffectRunner(harness.flow.submit, name="broken")

        async def broken(request: RequestAnalysis):
            raise RuntimeError("engine gone")

        harness.effects.register(
            RequestAnalysis,
            broken,
            on_error=lambda request, exc: AnalysisFailed(request.request_id, str(exc)),
        )
        harness.effects.register(RequestCoachLine, coach_line)
        harness.effects.register(RequestEngineMove, engine_move)
        async with harness:
            await harness.send(QuestionAsked("best"))
            self.assertTrue(harness.spoke("could not work that out"))
            await harness.send(QuestionAsked("hint"))
            self.assertTrue(harness.spoke("could not work that out"))


class ReviewFixTest(unittest.IsolatedAsyncioTestCase):
    """One test per defect found in the review pass, so none of them can come back."""

    async def test_why_was_that_bad_is_answered_not_refused(self):
        # The parser sends "why" with no move, and the move asked about is already
        # played, so the ask has to be built against the position *before* it.
        async with Harness(play_workflow(guard_enabled=False)) as harness:
            await harness.send(MoveSpoken("e2e4"))
            await harness.send(QuestionAsked("why"))
            self.assertTrue(harness.spoke("[coach_why] why"))
            self.assertFalse(harness.spoke("isn't possible"))

    async def test_why_before_any_move_says_so_honestly(self):
        async with Harness(play_workflow()) as harness:
            await harness.send(QuestionAsked("why"))
            self.assertTrue(harness.spoke("have not played a move yet"))

    async def test_an_undo_discards_the_engine_reply_it_was_thinking_about(self):
        # The engine answer arrives after the board was restored, so playing it
        # would push a move that is illegal in the position now on screen.
        held: list = []

        async def slow_engine(request: RequestEngineMove):
            held.append(request)

        workflow = play_workflow(guard_enabled=False)
        harness = Harness(workflow)
        harness.effects = EffectRunner(harness.flow.submit, name="slow")
        harness.effects.register(RequestEngineMove, slow_engine)
        harness.effects.register(RequestAnalysis, analyse)
        harness.effects.register(RequestCoachLine, coach_line)
        async with harness:
            await harness.send(MoveSpoken("e2e4"))
            self.assertEqual(len(held), 1)
            stale_reply = best_reply(chess.Board(held[0].fen))[1]
            await harness.send(Undo())
            self.assertEqual(workflow.state.fen, chess.Board().fen())
            await harness.send(EngineMoveReady(held[0].request_id, stale_reply))
            # The board is still the starting position: the stale move was dropped.
            self.assertEqual(workflow.state.fen, chess.Board().fen())
            self.assertFalse(harness.spoke("I played"))

    async def test_two_questions_in_a_row_are_both_answered(self):
        # A second question used to overwrite the first while the workflow was busy.
        async with Harness(play_workflow(ask_best_budget=-1, hint_budget=-1)) as harness:
            await harness.send(QuestionAsked("best"), QuestionAsked("hint"))
            self.assertTrue(harness.spoke("[coach_best] best"))
            self.assertTrue(harness.spoke("[coach_hint] hint"))

    async def test_a_failed_question_does_not_cost_the_child_a_turn(self):
        workflow = play_workflow(ask_best_budget=1)
        harness = Harness(workflow)
        harness.effects = EffectRunner(harness.flow.submit, name="broken")

        async def broken(request: RequestAnalysis):
            raise RuntimeError("engine gone")

        harness.effects.register(
            RequestAnalysis,
            broken,
            on_error=lambda request, exc: AnalysisFailed(request.request_id, str(exc)),
        )
        harness.effects.register(RequestCoachLine, coach_line)
        harness.effects.register(RequestEngineMove, engine_move)
        async with harness:
            await harness.send(QuestionAsked("best"))
            self.assertEqual(workflow.state.best_asks_used, 0)

    async def test_losing_reports_kid_lost_so_the_browser_does_not_celebrate(self):
        # Scholar's mate: white plays Qxf7 and the child, playing black, is mated.
        fen = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
        workflow = ChessWorkflow(policy=CoachPolicy(guard_enabled=False), mode="play")
        workflow.state.set_position(fen)
        workflow.state.kid_color = chess.BLACK
        async with Harness(workflow) as harness:
            await harness.send(MoveDragged("h5f7"))
            self.assertEqual(self.outcomes(harness), ["kid_lost"])
            self.assertTrue(harness.spoke("I won that one"))

    async def test_winning_reports_kid_won(self):
        fen = "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2"
        async with Harness(play_workflow(fen, guard_enabled=False)) as harness:
            await harness.send(MoveSpoken("d8h4"))
            self.assertEqual(self.outcomes(harness), ["kid_won"])

    async def test_a_stalemate_is_described_truthfully(self):
        # Qf2-f7 leaves the black king with no legal move and no check.
        fen = "7k/8/6K1/8/8/8/5Q2/8 w - - 0 1"
        async with Harness(play_workflow(fen, guard_enabled=False)) as harness:
            await harness.send(MoveDragged("f2f7"))
            self.assertTrue(harness.spoke("stalemate"))
            # The old wording claimed nobody could win, which is false here.
            self.assertFalse(harness.spoke("enough pieces"))
            self.assertEqual(self.outcomes(harness), ["draw"])

    async def test_a_draw_by_too_few_pieces_says_that_instead(self):
        # A bishop each: after a quiet king move neither side can force mate.
        fen = "7k/8/8/8/8/2b5/8/K1B5 w - - 0 1"
        async with Harness(play_workflow(fen, guard_enabled=False)) as harness:
            await harness.send(MoveDragged("a1a2"))
            self.assertTrue(harness.spoke("enough pieces"))
            self.assertFalse(harness.spoke("stalemate"))
            self.assertEqual(self.outcomes(harness), ["draw"])

    def outcomes(self, harness: Harness) -> list[str]:
        """Every ``outcome`` the browser was told for a finished game."""
        return [
            item.feedback.data["outcome"]
            for item in harness.outputs
            if item.feedback is not None and item.feedback.result == "game_over"
        ]


if __name__ == "__main__":
    unittest.main()
