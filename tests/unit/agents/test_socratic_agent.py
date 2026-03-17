"""
tests/unit/agents/test_socratic_agent.py
-----------------------------------------
Unit tests for SocraticAgent with a mocked LLM.
Verifies FlashcardDraft fields, list return type, and no settings dependency.

Note: SessionLocal is still imported in agents/socratic.py because
recreate_flashcard() and suggest_answer() are UI-driven and will be
decoupled in Phase 3. This test suite covers generate_flashcard() only.
"""

import json
import pytest
from unittest.mock import MagicMock

from agents.socratic import SocraticAgent, FlashcardDraft, FlashcardOutput, FlashcardItem, RubricItem


def _make_rubric_items():
    return [
        RubricItem(criterion="Accuracy", description="Answer matches the source."),
        RubricItem(criterion="Completeness", description="All key points covered."),
        RubricItem(criterion="Clarity", description="Response is unambiguous."),
    ]


def _make_flashcard_item(
    question="What is X?",
    answer="X is Y.",
    question_type="active_recall",
    complexity="medium",
) -> FlashcardItem:
    return FlashcardItem(
        question=question,
        answer=answer,
        question_type=question_type,
        rubric=_make_rubric_items(),
        suggested_complexity=complexity,
    )


def _make_output(cards=None) -> FlashcardOutput:
    if cards is None:
        cards = [_make_flashcard_item()]
    return FlashcardOutput(flashcards=cards)


def _install_chain(ag: SocraticAgent, output: FlashcardOutput, qtype: str = "active_recall") -> MagicMock:
    """Replace the named chain in the agent with a mock that returns the given output."""
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = output
    ag._chains[qtype] = mock_chain
    return mock_chain


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    chain = MagicMock()
    llm.with_structured_output.return_value = chain
    return llm, chain


@pytest.fixture
def agent(mock_llm):
    llm, chain = mock_llm
    return SocraticAgent(llm=llm)


class TestSocraticAgentReturnsFlashcardDraft:
    def test_returns_list_of_drafts(self, agent):
        _install_chain(agent, _make_output())
        result = agent.generate_flashcard(source_text="Some source text.")
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], FlashcardDraft)

    def test_draft_has_correct_fields(self, agent):
        _install_chain(agent, _make_output([
            _make_flashcard_item(question="Q?", answer="A.", question_type="fill_blank", complexity="simple")
        ]), qtype="fill_blank")
        result = agent.generate_flashcard(source_text="text", question_type="fill_blank")
        draft = result[0]
        assert draft.question == "Q?"
        assert draft.answer == "A."
        assert draft.question_type == "fill_blank"
        assert draft.suggested_complexity == "simple"

    def test_rubric_json_is_valid_list(self, agent):
        _install_chain(agent, _make_output())
        result = agent.generate_flashcard(source_text="text")
        rubric = json.loads(result[0].rubric_json)
        assert isinstance(rubric, list)
        assert len(rubric) == 3
        assert "criterion" in rubric[0]
        assert "description" in rubric[0]

    def test_multiple_drafts_returned(self, agent):
        _install_chain(agent, _make_output([
            _make_flashcard_item(question="Q1?"),
            _make_flashcard_item(question="Q2?"),
        ]))
        result = agent.generate_flashcard(source_text="text")
        assert len(result) == 2
        assert result[0].question == "Q1?"
        assert result[1].question == "Q2?"

    def test_empty_flashcards_returns_empty_list(self, agent):
        _install_chain(agent, FlashcardOutput(flashcards=[]))
        result = agent.generate_flashcard(source_text="text")
        assert result == []

    def test_question_type_from_caller_not_llm(self, agent):
        """question_type on FlashcardDraft always reflects the caller-requested type."""
        _install_chain(agent, _make_output([
            _make_flashcard_item(question_type="active_recall")  # LLM label
        ]), qtype="numerical")
        result = agent.generate_flashcard(source_text="text", question_type="numerical")
        assert result[0].question_type == "numerical"  # caller wins

    def test_no_settings_dependency_in_generate(self, agent):
        """generate_flashcard() must not read settings — side-effect free for unit tests."""
        # Verify by running without patching settings: if it raises, there's a hidden settings read
        _install_chain(agent, _make_output())
        result = agent.generate_flashcard(source_text="text")
        assert isinstance(result, list)


class TestSocraticAgentQuestionTypes:
    def test_unknown_question_type_falls_back_to_active_recall(self, agent):
        _install_chain(agent, _make_output())  # active_recall is the fallback
        result = agent.generate_flashcard(source_text="text", question_type="unknown_type")
        assert isinstance(result, list)
        assert len(result) == 1

    def test_all_supported_question_types_accepted(self, agent):
        for qtype in ("active_recall", "fill_blank", "short_answer", "long_answer", "numerical", "scenario"):
            _install_chain(agent, _make_output(), qtype=qtype)
            result = agent.generate_flashcard(source_text="text", question_type=qtype)
            assert isinstance(result, list), f"Failed for question_type={qtype}"
            assert len(result) == 1, f"Expected 1 draft for question_type={qtype}"


class TestSocraticAgentLegacyChunkArg:
    def test_source_text_resolved_from_chunk_page_content(self, agent):
        """Backward compat: when chunk has page_content and source_text is empty, use chunk."""
        mock_chain = MagicMock()
        captured_calls = []

        def side_effect(args):
            captured_calls.append(args["text"])
            return _make_output()

        mock_chain.invoke.side_effect = side_effect
        agent._chains["active_recall"] = mock_chain

        chunk = MagicMock()
        chunk.page_content = "resolved from chunk"
        agent.generate_flashcard(source_text="", chunk=chunk)
        assert len(captured_calls) == 1
        assert captured_calls[0] == "resolved from chunk"

    def test_source_text_takes_priority_over_chunk(self, agent):
        """If source_text is non-empty, chunk is ignored."""
        mock_chain = MagicMock()
        captured_calls = []

        def side_effect(args):
            captured_calls.append(args["text"])
            return _make_output()

        mock_chain.invoke.side_effect = side_effect
        agent._chains["active_recall"] = mock_chain

        chunk = MagicMock()
        chunk.page_content = "chunk content"
        agent.generate_flashcard(source_text="explicit source", chunk=chunk)
        assert len(captured_calls) == 1
        assert captured_calls[0] == "explicit source"

    def test_chunk_text_attr_fallback(self, agent):
        """Chunk with .text attribute (not .page_content) is also handled."""
        mock_chain = MagicMock()
        captured_calls = []

        def side_effect(args):
            captured_calls.append(args["text"])
            return _make_output()

        mock_chain.invoke.side_effect = side_effect
        agent._chains["active_recall"] = mock_chain

        chunk = MagicMock(spec=["text"])  # no page_content attr
        chunk.text = "from chunk.text"
        agent.generate_flashcard(source_text="", chunk=chunk)
        assert captured_calls[0] == "from chunk.text"
