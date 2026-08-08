"""Speech in, words out: the parser and the sentence builders.

No engine, no LLM, no network. Every case is checked against a real position, so a
reading that is not legal cannot pass by accident.
"""

import sys
import unittest
from pathlib import Path

import chess

MUAPPS = Path(__file__).resolve().parents[1] / "muapps"
if str(MUAPPS) not in sys.path:
    sys.path.insert(0, str(MUAPPS))

from chess_app.events import (
    ConfirmMove,
    MoveSpoken,
    NewGame,
    QuestionAsked,
    StopSession,
    TakeBackMove,
    UnclearInput,
    Undo,
)
from chess_app.move_parser import clean_words, interpret_speech, match_move, normalise
from chess_app.responses import (
    board_output,
    describe_score,
    plain_line,
    position_feedback,
    say_move,
    say_san,
    say_square,
    status_feedback,
)

from server.core.qa import Verdict

#: Knights on c3 and e3: both reach d5, so "knight to d five" is truly ambiguous.
TWO_KNIGHTS = "4k3/8/8/8/8/2N1N3/8/4K3 w - - 0 1"


def played(*sans: str) -> chess.Board:
    """A board reached by playing moves, so it also carries history."""
    board = chess.Board()
    for san in sans:
        board.push_san(san)
    return board


class NormaliseTests(unittest.TestCase):
    def test_homophones_are_repaired(self):
        self.assertEqual(normalise("Night to be four!"), "knight 2 b 4")
        self.assertEqual(normalise("horse eats ee five"), "knight takes e 5")

    def test_clean_words_leaves_english_alone(self):
        # This is the whole reason clean_words exists: normalise would turn "for"
        # into "4" and "to" into "2", hiding the meaning.
        self.assertEqual(clean_words("Best move for black?"), "best move for black")
        self.assertEqual(normalise("Best move for black?"), "best move 4 black")

    def test_empty_speech_is_unclear(self):
        self.assertIsInstance(interpret_speech("", chess.Board()), UnclearInput)
        self.assertIsInstance(interpret_speech("...", chess.Board()), UnclearInput)


class MoveSpeechTests(unittest.TestCase):
    def setUp(self):
        self.board = chess.Board()

    def test_a_plain_pawn_move(self):
        event = interpret_speech("e4", self.board)
        self.assertIsInstance(event, MoveSpoken)
        self.assertEqual(event.uci, "e2e4")

    def test_a_split_up_square(self):
        event = interpret_speech("pawn to ee four", self.board)
        self.assertEqual(event.uci, "e2e4")

    def test_a_misheard_knight(self):
        event = interpret_speech("night to eff three", self.board)
        self.assertEqual(event.uci, "g1f3")

    def test_two_knights_reaching_one_square_stays_ambiguous(self):
        # Knights on c3 and e3 both reach d5, so the parser must not guess.
        board = chess.Board(TWO_KNIGHTS)
        self.assertEqual(sorted(match_move(board, normalise("knight to d five"))), ["c3d5", "e3d5"])
        self.assertIsInstance(interpret_speech("knight to d five", board), UnclearInput)

    def test_naming_the_origin_picks_one_of_two_knights(self):
        board = chess.Board(TWO_KNIGHTS)
        self.assertEqual(match_move(board, normalise("see three to d five")), ["c3d5"])
        self.assertEqual(interpret_speech("see three to d five", board).uci, "c3d5")

    def test_a_capture_must_really_capture(self):
        board = played("e4", "d5")
        self.assertEqual(match_move(board, normalise("pawn takes d five")), ["e4d5"])
        self.assertEqual(match_move(board, normalise("knight takes d five")), [])

    def test_castling_short_and_long(self):
        board = played("e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5")
        event = interpret_speech("castle", board)
        self.assertIsInstance(event, MoveSpoken)
        self.assertEqual(event.uci, "e1g1")
        # Only kingside is legal here, so asking to castle long must not silently
        # play the wrong move.
        self.assertIsInstance(interpret_speech("castle long", board), UnclearInput)

    def test_castling_long_when_it_is_legal(self):
        board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
        self.assertEqual(interpret_speech("castle long", board).uci, "e1c1")
        self.assertEqual(interpret_speech("castle short", board).uci, "e1g1")

    def test_an_illegal_move_is_unclear_not_played(self):
        self.assertIsInstance(interpret_speech("queen to h five", chess.Board()), UnclearInput)


class PromotionSpeechTests(unittest.TestCase):
    def setUp(self):
        # A white pawn on a7 with nothing in the way: a8 is a promotion square.
        self.board = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")

    def test_a_bare_promotion_square_promotes_to_a_queen(self):
        self.assertEqual(interpret_speech("a eight", self.board).uci, "a7a8q")

    def test_promoting_to_a_rook(self):
        # "to" normalises to "2" and "rook" is the prize, not the mover — both used
        # to break this phrase.
        self.assertEqual(interpret_speech("promote to a rook", self.board).uci, "a7a8r")

    def test_promoting_to_a_bishop_on_a_named_square(self):
        self.assertEqual(interpret_speech("a eight promote to a bishop", self.board).uci, "a7a8b")

    def test_promoting_to_a_knight(self):
        self.assertEqual(interpret_speech("promote to a knight", self.board).uci, "a7a8n")


class CommandSpeechTests(unittest.TestCase):
    def setUp(self):
        self.board = chess.Board()

    def check(self, said: str, expected):
        self.assertIsInstance(interpret_speech(said, self.board), expected)

    def test_each_command_word(self):
        self.check("stop", StopSession)
        self.check("new game", NewGame)
        self.check("undo", Undo)
        self.check("take it back", TakeBackMove)
        self.check("yes", ConfirmMove)
        self.check("no wait", TakeBackMove)

    def test_a_command_may_carry_trailing_words(self):
        self.check("new game please", NewGame)

    def test_a_command_wins_over_a_move_reading(self):
        # "undo" contains no square, but the ordering matters for phrases that do.
        self.check("go back a move", Undo)


class QuestionSpeechTests(unittest.TestCase):
    def setUp(self):
        self.board = played("e4", "e5", "Nf3", "Nc6")

    def ask(self, said: str) -> QuestionAsked:
        event = interpret_speech(said, self.board)
        self.assertIsInstance(event, QuestionAsked)
        return event

    def test_best_move(self):
        self.assertEqual(self.ask("what's the best move").kind, "best")

    def test_best_move_for_a_named_side(self):
        event = self.ask("what's the best move for black")
        self.assertEqual((event.kind, event.side), ("best", "black"))

    def test_what_would_white_play(self):
        event = self.ask("what would white play")
        self.assertEqual((event.kind, event.side), ("best", "white"))

    def test_a_possessive_side_is_understood(self):
        self.assertEqual(self.ask("what is black's best move").side, "black")

    def test_what_if_carries_the_move_it_asks_about(self):
        event = self.ask("what if I play knight takes e five")
        self.assertEqual((event.kind, event.action_text), ("whatif", "f3e5"))

    def test_a_what_if_about_an_impossible_move_degrades_to_a_hint(self):
        # Better to nudge than to answer a question about a move that cannot be
        # played.
        self.assertEqual(self.ask("what if I play queen takes a eight").kind, "hint")

    def test_why_needs_no_move(self):
        event = self.ask("why was that bad")
        self.assertEqual((event.kind, event.action_text), ("why", ""))

    def test_hint_and_explain(self):
        self.assertEqual(self.ask("give me a hint").kind, "hint")
        self.assertEqual(self.ask("who is winning").kind, "explain")


class SaySquareTests(unittest.TestCase):
    def test_files_are_spoken_as_words(self):
        self.assertEqual(say_square("f7"), "eff 7")
        self.assertEqual(say_square("a1"), "a 1")
        self.assertEqual(say_square("h8"), "aitch 8")

    def test_nonsense_is_returned_unchanged(self):
        self.assertEqual(say_square("z9"), "z9")
        self.assertEqual(say_square(""), "")


class SaySanTests(unittest.TestCase):
    def test_a_quiet_move(self):
        self.assertEqual(say_san("Nf3"), "knight to eff 3")

    def test_a_capture_with_check(self):
        self.assertEqual(say_san("Bxf7+"), "bishop takes eff 7, check")

    def test_checkmate(self):
        self.assertEqual(say_san("Qh4#"), "queen to aitch 4, checkmate")

    def test_castling(self):
        self.assertEqual(say_san("O-O"), "castles short")
        self.assertEqual(say_san("O-O-O"), "castles long")
        self.assertEqual(say_san("O-O#"), "castles short, checkmate")

    def test_a_pawn_move_names_no_piece(self):
        self.assertEqual(say_san("e4"), "pawn to ee 4")
        # SAN always names the capturing pawn's file, and the child should hear it.
        self.assertEqual(say_san("exd5"), "pawn from ee takes dee 5")

    def test_promotion(self):
        self.assertEqual(say_san("a8=Q"), "pawn to a 8, promoting to a queen")
        self.assertEqual(say_san("a8=N+"), "pawn to a 8, promoting to a knight, check")

    def test_a_file_disambiguation_is_spoken_not_mangled(self):
        # Before the fix this said "knight to ge7", a square that does not exist —
        # and this is exactly when the child needs to hear which knight moved.
        self.assertEqual(say_san("Nge7"), "knight from gee to ee 7")

    def test_a_rank_disambiguation(self):
        self.assertEqual(say_san("R1e2"), "rook from row 1 to ee 2")

    def test_a_full_square_disambiguation(self):
        self.assertEqual(say_san("Qh4xe1"), "queen from aitch 4 takes ee 1")

    def test_a_disambiguated_capture(self):
        self.assertEqual(say_san("Nexd4"), "knight from ee takes dee 4")


class SayMoveTests(unittest.TestCase):
    def test_a_legal_move_is_described(self):
        board = played("e4", "e5", "Bc4", "Nc6")
        self.assertEqual(say_move(board, "c4f7"), "bishop takes eff 7, check")

    def test_an_illegal_or_broken_move_says_nothing(self):
        board = chess.Board()
        self.assertEqual(say_move(board, "e2e5"), "")
        self.assertEqual(say_move(board, "banana"), "")
        self.assertEqual(say_move(board, ""), "")


class DescribeScoreTests(unittest.TestCase):
    def test_no_numbers_ever_appear(self):
        for score in (-2000, -400, -100, 0, 100, 400, 2000):
            said = describe_score(score)
            self.assertTrue(said)
            self.assertFalse(any(char.isdigit() for char in said), said)

    def test_the_bands_read_the_right_way_round(self):
        self.assertEqual(describe_score(900), "You are winning.")
        self.assertEqual(describe_score(0), "The game is about level.")
        self.assertIn("losing", describe_score(-900))


class PlainLineTests(unittest.TestCase):
    def setUp(self):
        self.board = played("e4", "e5", "Bc4", "Nc6")

    def line(self, kind: str, verdict: Verdict, side: str = "") -> str:
        from chess_app.events import ChessAsk

        extra = {"side": side} if side else {}
        ask = ChessAsk(kind=kind, snapshot=self.board.fen(), extra=extra)
        return plain_line(self.board, ask, verdict)

    def test_best_names_the_move_in_words(self):
        said = self.line("best", Verdict(best_action="c4f7"))
        self.assertEqual(said, "I would play bishop takes eff 7, check.")

    def test_asking_about_the_other_side_is_flagged(self):
        said = self.line("best", Verdict(best_action="c4f7"), side="black")
        self.assertTrue(said.startswith("If it were black's turn,"))

    def test_a_hint_never_names_the_move(self):
        said = self.line("hint", Verdict(best_action="c4f7"))
        self.assertNotIn("eff 7", said)
        self.assertNotIn("bishop", said)

    def test_what_if_reports_the_reply(self):
        verdict = Verdict(detail={"san": "Bxf7+", "reply_san": "Kxf7"})
        said = self.line("whatif", verdict)
        self.assertEqual(said, "After bishop takes eff 7, check, I would answer king takes eff 7.")

    def test_explain_talks_about_the_score_in_words(self):
        self.assertEqual(self.line("explain", Verdict(score=900)), "You are winning.")

    def test_an_unknown_kind_says_nothing(self):
        self.assertEqual(self.line("check", Verdict()), "")


class BrowserMessageTests(unittest.TestCase):
    def test_position_feedback_carries_the_fen_and_the_legal_moves(self):
        board = chess.Board()
        legal = tuple(move.uci() for move in board.legal_moves)
        message = position_feedback(board.fen(), legal=legal, arrow="e2e4")
        self.assertEqual(message.name, "chess_position")
        self.assertEqual(message.data["fen"], board.fen())
        self.assertEqual(message.data["arrow"], "e2e4")
        self.assertEqual(len(message.data["legal"]), 20)

    def test_status_feedback_passes_extra_fields_through(self):
        message = status_feedback("warned", uci="d8h4")
        self.assertEqual((message.name, message.result), ("chess_status", "warned"))
        self.assertEqual(message.data["uci"], "d8h4")

    def test_board_output_always_ships_a_fresh_position(self):
        from chess_app.domain import ChessState

        state = ChessState()
        state.board.push_san("e4")
        packet = board_output(state, "your turn", arrow="e7e5")
        self.assertEqual(packet.messages, ["your turn"])
        self.assertEqual(packet.feedback.data["fen"], state.board.fen())
        self.assertEqual(packet.feedback.data["arrow"], "e7e5")
        # The browser is told what is legal, so it never needs chess rules itself.
        self.assertIn("e7e5", packet.feedback.data["legal"])

    def test_board_output_drops_empty_messages(self):
        from chess_app.domain import ChessState

        packet = board_output(ChessState(), "", "something", "")
        self.assertEqual(packet.messages, ["something"])


if __name__ == "__main__":
    unittest.main()
