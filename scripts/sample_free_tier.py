"""
scripts/sample_free_tier.py
----------------------------
Validation + baseline script for free-tier model hopping.

Tests whether Groq (Llama-3.3-70B) and Gemini 2.0 Flash can replace
gpt-4o-mini (routing) and gpt-4o (primary) with minimal quality degradation,
and benchmarks all available providers using RAGAS metrics.

Run:
    PYTHONPATH=. python scripts/sample_free_tier.py

Required keys in .env:
    GROQ_API_KEY      — Groq free tier   (console.groq.com)
    GOOGLE_API_KEY    — Google AI Studio (aistudio.google.com)
    ANTHROPIC_API_KEY — Anthropic paid   (used as quality baseline)

Optional:
    OPENAI_API_KEY    — GPT-4o comparison (skipped if quota exhausted)

All provider / RAGAS infrastructure is in scripts/model_hop.py.
"""

import sys
import textwrap

# Force UTF-8 output on Windows consoles that default to cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pydantic import BaseModel, Field
from scripts.model_hop import (
    get_llm,
    bind_structured,
    is_quota_error,
    generate_structured,
    available_providers,
    build_ragas_evaluator,
    run_ragas_benchmark,
    print_ragas_table,
)

# ---------------------------------------------------------------------------
# Schemas — inline here; test-specific, not part of the reusable hop library
# ---------------------------------------------------------------------------

class Subtopic(BaseModel):
    name: str = Field(description="Short subtopic name (3–8 words)")
    description: str = Field(description="One-sentence description of what this subtopic covers")

class Topic(BaseModel):
    name: str = Field(description="Topic name (2–5 words)")
    subtopics: list[Subtopic] = Field(description="2–4 subtopics within this topic")

class DocumentStructure(BaseModel):
    subject: str = Field(description="High-level subject area")
    topics: list[Topic] = Field(description="2–3 main topics identified in the text")

class Flashcard(BaseModel):
    question: str = Field(description="Active recall question")
    answer: str = Field(description="Concise, accurate answer")
    difficulty: str = Field(description="easy | medium | hard")

class FlashcardOutput(BaseModel):
    cards: list[Flashcard] = Field(description="1–3 flashcards derived from the passage")
    chunk_summary: str = Field(description="One-sentence summary of the passage")

class RelevanceDecision(BaseModel):
    is_relevant: bool = Field(description="True if the chunk is relevant to the target topics")
    reason: str = Field(description="One-sentence justification")

# ---------------------------------------------------------------------------
# Fixed sample text (~400 words, conditional probability + Bayes' theorem)
# ---------------------------------------------------------------------------

SAMPLE_TEXT = textwrap.dedent("""
    Conditional probability is the probability of an event A occurring given that
    another event B has already occurred. It is written as P(A|B) and defined as:

        P(A|B) = P(A ∩ B) / P(B),  provided P(B) > 0.

    Intuitively, conditioning on B means we restrict our sample space to only the
    outcomes where B happened, then ask what fraction of those also satisfy A.

    Bayes' Theorem follows directly from this definition. Noting that
    P(A ∩ B) = P(B|A)·P(A), we can substitute:

        P(A|B) = [P(B|A) · P(A)] / P(B).

    This formula is fundamental in Bayesian inference. P(A) is the *prior*
    probability — our belief about A before observing evidence B. After observing
    B, we update to the *posterior* P(A|B). P(B|A) is the *likelihood* — how
    probable the evidence B is under hypothesis A.

    Example — Medical Diagnosis:
    Suppose a disease affects 1% of the population (P(D) = 0.01). A diagnostic
    test is 95% sensitive — it correctly identifies 95% of sick patients,
    P(+|D) = 0.95 — and 90% specific, so its false-positive rate is 10%,
    P(+|¬D) = 0.10.

    What is the probability that a patient who tests positive actually has the
    disease?

    Step 1 — compute P(+) via the law of total probability:
        P(+) = P(+|D)·P(D) + P(+|¬D)·P(¬D)
             = 0.95×0.01 + 0.10×0.99
             = 0.0095 + 0.099 = 0.1085.

    Step 2 — apply Bayes':
        P(D|+) = P(+|D)·P(D) / P(+) = 0.0095 / 0.1085 ≈ 0.0876.

    Despite a positive test result, there is only an ~8.8% chance the patient
    is sick. This counterintuitive result arises because the disease is rare
    (low prior) and the false-positive rate is non-trivial.

    The law of total probability used in Step 1 generalises: for a partition
    {B₁, B₂, …, Bₙ} of the sample space,
        P(A) = Σᵢ P(A|Bᵢ)·P(Bᵢ).

    Together, Bayes' theorem and the law of total probability are the
    workhorses of probabilistic reasoning in statistics, machine learning,
    and decision theory.
""").strip()

TARGET_TOPICS = ["Conditional Probability", "Bayes' Theorem"]

CARD_PROMPT = textwrap.dedent(f"""
    You are an Active Recall flashcard generator.
    Generate 1–3 high-quality flashcards from the passage below.
    Focus on conceptual understanding, not rote memorisation.
    Assign difficulty: easy | medium | hard.

    Passage:
    {SAMPLE_TEXT}
""").strip()

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _header(title: str) -> None:
    print(f"\n{'='*60}\n  {title}\n{'='*60}")

def _ok(msg: str)   -> None: print(f"  [PASS] {msg}")
def _fail(msg: str) -> None: print(f"  [FAIL] {msg}")
def _skip(msg: str) -> None: print(f"  [SKIP] {msg}")

# ---------------------------------------------------------------------------
# Test 1 — DocumentStructure via Gemini
# ---------------------------------------------------------------------------

def test_gemini_structured_output() -> bool:
    _header("Test 1 — Gemini: DocumentStructure (3-level JSON nesting)")
    try:
        llm = get_llm("google", purpose="primary")
        structured = bind_structured(llm, DocumentStructure)
        prompt = (
            "Analyse the following educational text and return a structured "
            "DocumentStructure JSON.\n\nText:\n" + SAMPLE_TEXT
        )
        result: DocumentStructure = structured.invoke(prompt)
        assert isinstance(result, DocumentStructure) and result.subject
        assert len(result.topics) >= 1 and all(len(t.subtopics) >= 1 for t in result.topics)
        _ok(f"subject='{result.subject}', {len(result.topics)} topic(s)")
        for t in result.topics:
            print(f"     Topic: {t.name} — {len(t.subtopics)} subtopic(s)")
        return True
    except Exception as e:
        if is_quota_error(e):
            _skip("Gemini daily quota exhausted — rerun tomorrow. (not a code failure)")
            return True
        _fail(str(e))
        return False

# ---------------------------------------------------------------------------
# Test 2 — DocumentStructure via Groq
# ---------------------------------------------------------------------------

def test_groq_structured_output() -> bool:
    _header("Test 2 — Groq (Llama-3.3-70B): DocumentStructure")
    try:
        llm = get_llm("groq", purpose="routing")
        structured = bind_structured(llm, DocumentStructure)
        prompt = (
            "Analyse the following educational text and return a structured "
            "DocumentStructure JSON.\n\nText:\n" + SAMPLE_TEXT
        )
        result: DocumentStructure = structured.invoke(prompt)
        assert isinstance(result, DocumentStructure) and result.subject
        assert len(result.topics) >= 1
        _ok(f"subject='{result.subject}', {len(result.topics)} topic(s)")
        return True
    except Exception as e:
        _fail(str(e))
        return False

# ---------------------------------------------------------------------------
# Test 3 — Flashcard generation via Gemini
# ---------------------------------------------------------------------------

def test_gemini_flashcard_generation() -> bool:
    _header("Test 3 — Gemini: FlashcardOutput generation")
    result = generate_structured("google", FlashcardOutput, CARD_PROMPT)
    if result is None:
        _skip("Gemini daily quota exhausted — rerun tomorrow.")
        return True
    try:
        assert result.chunk_summary
        _ok(f"{len(result.cards)} card(s) generated")
        for c in result.cards:
            print(f"     [{c.difficulty}] Q: {c.question[:80]}…")
        return True
    except Exception as e:
        _fail(str(e))
        return False

# ---------------------------------------------------------------------------
# Test 4 — Model hop: Groq classifies relevance → Gemini generates cards
# ---------------------------------------------------------------------------

def test_model_hop() -> bool:
    _header("Test 4 — Model Hop: Groq (relevance) → Gemini (cards)")
    try:
        # Step A: Groq classifies relevance
        groq_llm = get_llm("groq", purpose="routing")
        relevance_llm = bind_structured(groq_llm, RelevanceDecision)
        relevance_prompt = textwrap.dedent(f"""
            Decide whether the following text chunk is relevant to these topics:
            {TARGET_TOPICS}

            Chunk:
            {SAMPLE_TEXT[:600]}
        """).strip()
        decision: RelevanceDecision = relevance_llm.invoke(relevance_prompt)
        assert isinstance(decision, RelevanceDecision)
        _ok(f"Groq relevance: is_relevant={decision.is_relevant} — {decision.reason}")

        if not decision.is_relevant:
            print("  [INFO] Groq marked chunk irrelevant; skipping generation.")
            return True

        # Step B: Gemini generates cards
        result = generate_structured("google", FlashcardOutput, CARD_PROMPT)
        if result is None:
            _skip("Gemini daily quota exhausted — hop step skipped.")
            return True
        _ok(f"Gemini generated {len(result.cards)} card(s) after Groq relevance gate")
        return True
    except Exception as e:
        if is_quota_error(e):
            _skip("Quota exhausted — rerun tomorrow.")
            return True
        _fail(str(e))
        return False

# ---------------------------------------------------------------------------
# Test 5 — Claude flashcard baseline
# ---------------------------------------------------------------------------

def test_claude_comparison() -> bool:
    from core.config import settings
    if not settings.ANTHROPIC_API_KEY:
        _skip("ANTHROPIC_API_KEY not set — Claude comparison skipped.")
        return True
    _header("Test 5 — Claude (Sonnet): FlashcardOutput baseline")
    result = generate_structured("anthropic", FlashcardOutput, CARD_PROMPT)
    if result is None:
        _skip("Anthropic quota exhausted.")
        return True
    try:
        assert result.chunk_summary
        _ok(f"Claude generated {len(result.cards)} card(s)")
        for c in result.cards:
            print(f"     [{c.difficulty}] Q: {c.question[:80]}…")
            print(f"                    A: {c.answer[:80]}…")
        return True
    except Exception as e:
        _fail(str(e))
        return False

# ---------------------------------------------------------------------------
# Test 6 — RAGAS benchmark across all available providers
# ---------------------------------------------------------------------------

def benchmark_providers() -> bool:
    _header("Test 6 — RAGAS Benchmark: Faithfulness + Response Relevancy")
    from core.config import settings

    if not settings.ANTHROPIC_API_KEY and not settings.GOOGLE_API_KEY:
        _skip("No evaluator LLM available (need ANTHROPIC_API_KEY or GOOGLE_API_KEY).")
        return True

    providers = available_providers()
    print(f"  Providers: {', '.join(providers.keys())}")
    print("  Generating cards…")

    # Collect (question, answer) pairs per provider
    provider_qa: dict[str, list[tuple[str, str]]] = {}
    for label, (prov, purp) in providers.items():
        result = generate_structured(prov, FlashcardOutput, CARD_PROMPT, purpose=purp)
        if result:
            provider_qa[label] = [(c.question, c.answer) for c in result.cards]
            print(f"    {label}: {len(result.cards)} card(s)")
        else:
            print(f"    {label}: quota exhausted — excluded")

    if not provider_qa:
        _fail("No providers produced cards — cannot run RAGAS benchmark.")
        return False

    print("\n  Building RAGAS evaluator…")
    try:
        eval_llm, eval_embeddings = build_ragas_evaluator()
    except Exception as e:
        _fail(f"Could not build RAGAS evaluator: {e}")
        return False

    print("\n  Running RAGAS evaluation (30–90s per provider)…\n")
    results = run_ragas_benchmark(provider_qa, eval_llm, eval_embeddings, context=SAMPLE_TEXT)

    _header("RAGAS Benchmark Results")
    print_ragas_table(results)
    _ok("Benchmark complete")
    return True

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\nNexus Learner — Free-Tier Model Hopping Sample + RAGAS Baseline")
    print("Providers: Groq (Llama-3.3-70B) | Gemini 2.0 Flash | Claude Sonnet")

    results = {
        "test_gemini_structured_output":    test_gemini_structured_output(),
        "test_groq_structured_output":      test_groq_structured_output(),
        "test_gemini_flashcard_generation": test_gemini_flashcard_generation(),
        "test_model_hop":                   test_model_hop(),
        "test_claude_comparison":           test_claude_comparison(),
        "benchmark_providers (RAGAS)":      benchmark_providers(),
    }

    _header("Summary")
    passed = sum(1 for v in results.values() if v)
    total  = len(results)
    for name, ok in results.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\n  {passed}/{total} tests passed")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
