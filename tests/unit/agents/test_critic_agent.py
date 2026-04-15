"""
tests/unit/agents/test_critic_agent.py
----------------------------------------
Unit tests for CriticAgent with a mocked LLM.
Verifies CriticResult fields, auto-reject logic, and zero DB dependency.
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from agents.critic import CriticAgent, CriticResult, RubricEvaluation


def _make_eval(
    accuracy=4, logic=4, grounding=4, clarity=4,
    feedback="ok", complexity="medium"
) -> RubricEvaluation:
    return RubricEvaluation(
        accuracy_score=accuracy,
        logic_score=logic,
        grounding_score=grounding,
        clarity_score=clarity,
        feedback=feedback,
        suggested_complexity=complexity,
    )


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    chain = MagicMock()
    llm.with_structured_output.return_value = chain
    return llm, chain


@pytest.fixture
def agent(mock_llm):
    llm, chain = mock_llm
    with patch("agents.critic.call_structured_chain") as mock_call:
        mock_call.return_value = _make_eval()
        ag = CriticAgent(llm=llm)
        ag._mock_call = mock_call
    return ag, mock_llm[1]


class TestCriticAgentReturnsResult:
    def test_returns_critic_result(self, agent):
        ag, _ = agent
        with patch("agents.critic.call_structured_chain") as mock_call:
            mock_call.return_value = _make_eval(accuracy=4, logic=4, grounding=4, clarity=4)
            result = ag.evaluate_flashcard(
                source_text="The sky is blue.",
                question="What color is the sky?",
                answer="Blue.",
            )
        assert isinstance(result, CriticResult)

    def test_aggregate_score_is_mean_of_four(self, agent):
        ag, _ = agent
        with patch("agents.critic.call_structured_chain") as mock_call:
            mock_call.return_value = _make_eval(accuracy=4, logic=4, grounding=4, clarity=4)
            result = ag.evaluate_flashcard("text", "q", "a")
        assert result.aggregate_score == 4

    def test_rubric_scores_json_is_valid(self, agent):
        ag, _ = agent
        with patch("agents.critic.call_structured_chain") as mock_call:
            mock_call.return_value = _make_eval(accuracy=3, logic=2, grounding=4, clarity=4)
            result = ag.evaluate_flashcard("text", "q", "a")
        scores = json.loads(result.rubric_scores_json)
        assert scores["accuracy"] == 3
        assert scores["logic"] == 2

    def test_complexity_normalised_when_unknown(self, agent):
        ag, _ = agent
        with patch("agents.critic.call_structured_chain") as mock_call:
            mock_call.return_value = _make_eval(complexity="hard")  # unknown value
            result = ag.evaluate_flashcard("text", "q", "a")
        assert result.suggested_complexity == "medium"

    def test_no_db_import_in_agent(self):
        import agents.critic as critic_module
        assert not hasattr(critic_module, "SessionLocal"), (
            "CriticAgent must not import SessionLocal — DB knowledge must live in the workflow node"
        )


class TestCriticAgentAutoReject:
    def test_auto_reject_when_grounding_lt_2(self, agent):
        ag, _ = agent
        with patch("agents.critic.call_structured_chain") as mock_call:
            mock_call.return_value = _make_eval(grounding=1)
            with patch("agents.critic.settings") as mock_settings:
                mock_settings.AUTO_ACCEPT_CONTENT = False
                mock_settings.MODEL_HOP_ENABLED = False
                mock_settings.CRITIC_REJECT_MIN_SCORE = 2
                result = ag.evaluate_flashcard("text", "q", "a")
        assert result.should_reject is True
        assert "grounding" in result.reject_reason

    def test_auto_reject_when_clarity_lt_2(self, agent):
        ag, _ = agent
        with patch("agents.critic.call_structured_chain") as mock_call:
            mock_call.return_value = _make_eval(clarity=1)
            with patch("agents.critic.settings") as mock_settings:
                mock_settings.AUTO_ACCEPT_CONTENT = False
                mock_settings.MODEL_HOP_ENABLED = False
                mock_settings.CRITIC_REJECT_MIN_SCORE = 2
                result = ag.evaluate_flashcard("text", "q", "a")
        assert result.should_reject is True
        assert "clarity" in result.reject_reason

    def test_no_reject_when_grounding_gte_2(self, agent):
        ag, _ = agent
        with patch("agents.critic.call_structured_chain") as mock_call:
            mock_call.return_value = _make_eval(grounding=2, clarity=3)
            with patch("agents.critic.settings") as mock_settings:
                mock_settings.AUTO_ACCEPT_CONTENT = False
                mock_settings.MODEL_HOP_ENABLED = False
                mock_settings.CRITIC_REJECT_MIN_SCORE = 2
                result = ag.evaluate_flashcard("text", "q", "a")
        assert result.should_reject is False

    def test_auto_accept_bypasses_auto_reject(self, agent):
        ag, _ = agent
        with patch("agents.critic.call_structured_chain") as mock_call:
            mock_call.return_value = _make_eval(grounding=1)
            with patch("agents.critic.settings") as mock_settings:
                mock_settings.AUTO_ACCEPT_CONTENT = True
                mock_settings.MODEL_HOP_ENABLED = False
                mock_settings.CRITIC_REJECT_MIN_SCORE = 2
                result = ag.evaluate_flashcard("text", "q", "a")
        assert result.should_reject is False


class TestCriticAgentErrorHandling:
    def test_llm_failure_returns_error_result(self, agent):
        ag, _ = agent
        with patch("agents.critic.call_structured_chain") as mock_call:
            mock_call.side_effect = Exception("LLM down")
            result = ag.evaluate_flashcard("text", "q", "a")
        assert result.error is not None
        assert "LLM down" in result.error
        assert result.should_reject is False  # don't reject on eval failure
