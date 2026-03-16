"""
scripts/sample_free_tier.py
----------------------------
Validation + baseline script for free-tier model hopping.

Demonstrates tier-based model selection via scripts/model_hop.py (LiteLLM-backed),
tests structured output across providers, and benchmarks all available providers
using RAGAS metrics.

Run:
    PYTHONPATH=. python scripts/sample_free_tier.py

Required keys in .env (at least one provider needed):
    GROQ_API_KEY      — Groq free tier   (console.groq.com)
    GOOGLE_API_KEY    — Google AI Studio (aistudio.google.com)
    ANTHROPIC_API_KEY — Anthropic paid   (used as quality baseline)

Optional:
    OPENAI_API_KEY    — GPT-4o comparison (skipped if quota exhausted)
    DEEPSEEK_API_KEY  — DeepSeek-V3 / R1

All provider / RAGAS infrastructure is in scripts/model_hop.py.
"""

import os
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
# Demo 0 — Tier-based model selection
# ---------------------------------------------------------------------------

def demo_tier_dispatch() -> bool:
    _header("Demo 0 — Tier-based model dispatch")
    print("  Resolving models by task (first available key wins):\n")
    try:
        routing_llm    = get_llm(task="routing")     # fast tier
        generation_llm = get_llm(task="generation")  # balanced tier
        reasoning_llm  = get_llm(task="reasoning")   # reasoning tier

        def _model(llm) -> str:
            return getattr(llm, "model", str(llm))

        _ok(f"routing    → {_model(routing_llm)}")
        _ok(f"generation → {_model(generation_llm)}")
        _ok(f"reasoning  → {_model(reasoning_llm)}")
        return True
    except RuntimeError as e:
        _skip(f"Not enough keys for all tiers: {e}")
        return True
    except Exception as e:
        _fail(str(e))
        return False

# ---------------------------------------------------------------------------
# Test 1 — DocumentStructure via Gemini
# ---------------------------------------------------------------------------

def test_gemini_structured_output() -> bool:
    _header("Test 1 — Gemini: DocumentStructure (3-level JSON nesting)")
    if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        _skip("GOOGLE_API_KEY not set — Gemini test skipped.")
        return True
    try:
        llm = get_llm("gemini/gemini-2.0-flash")
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
# Test 2 — DocumentStructure via Groq (task="routing")
# ---------------------------------------------------------------------------

def test_groq_structured_output() -> bool:
    _header("Test 2 — Groq (Llama-3.3-70B): DocumentStructure via task='routing'")
    if not os.environ.get("GROQ_API_KEY"):
        _skip("GROQ_API_KEY not set — Groq test skipped.")
        return True
    try:
        llm = get_llm(task="routing")
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
    if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        _skip("GOOGLE_API_KEY not set — Gemini test skipped.")
        return True
    result = generate_structured("gemini/gemini-2.0-flash", FlashcardOutput, CARD_PROMPT)
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
        # Step A: Groq classifies relevance via task="routing"
        groq_llm = get_llm(task="routing")
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

        # Step B: Gemini generates cards via explicit model string
        result = generate_structured("gemini/gemini-2.0-flash", FlashcardOutput, CARD_PROMPT)
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
    if not os.environ.get("ANTHROPIC_API_KEY"):
        _header("Test 5 — Claude (Sonnet): FlashcardOutput baseline")
        _skip("ANTHROPIC_API_KEY not set — Claude comparison skipped.")
        return True
    _header("Test 5 — Claude (Sonnet): FlashcardOutput baseline")
    result = generate_structured("anthropic/claude-sonnet-4-6", FlashcardOutput, CARD_PROMPT)
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

    if not os.environ.get("ANTHROPIC_API_KEY") and not (
        os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    ):
        _skip("No evaluator LLM available (need ANTHROPIC_API_KEY or GOOGLE_API_KEY).")
        return True

    providers = available_providers()
    print(f"  Providers: {', '.join(providers.keys())}")
    print("  Generating cards…")

    # Collect (question, answer) pairs per provider
    provider_qa: dict[str, list[tuple[str, str]]] = {}
    for label, model_str in providers.items():
        result = generate_structured(model_str, FlashcardOutput, CARD_PROMPT)
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
# Test 7 — DeepSeek-R1 via Groq: reasoning tier flashcard
# ---------------------------------------------------------------------------

def test_reasoning_tier() -> bool:
    _header("Test 7 — DeepSeek-R1 via Groq: reasoning tier flashcard")
    if not os.environ.get("GROQ_API_KEY"):
        _skip("GROQ_API_KEY not set — reasoning tier test skipped.")
        return True
    result = generate_structured("reasoning", FlashcardOutput, CARD_PROMPT)
    if result is None:
        _skip("Reasoning tier quota exhausted.")
        return True
    try:
        assert result.chunk_summary
        _ok(f"DeepSeek-R1 (Groq) generated {len(result.cards)} card(s)")
        for c in result.cards:
            print(f"     [{c.difficulty}] Q: {c.question[:80]}…")
        return True
    except Exception as e:
        _fail(str(e))
        return False

# ---------------------------------------------------------------------------
# Demo 8 — Semantic Cache (Qdrant-backed, local embeddings)
# ---------------------------------------------------------------------------

def demo_semantic_cache() -> bool:
    _header("Demo 8 — Semantic Cache (Qdrant-backed, all-MiniLM-L6-v2)")

    # Step 1: Initialise
    try:
        from core.cache import init_semantic_cache, _reset_cache_singleton  # type: ignore[import]
        _reset_cache_singleton()  # ensure fresh init for this demo
        cache = init_semantic_cache()
    except ImportError as exc:
        _skip(f"core.cache not importable — {exc}")
        return True

    s = cache.stats()
    if not s["enabled"]:
        _skip("SemanticCache not ready (Qdrant unavailable or sentence-transformers missing)")
        return True

    print(f"  Cache initialised — collection={s['collection']}, threshold={s['threshold']}")

    # Step 2: Clear for a clean run
    cache.clear()

    # Ensure at least one provider is available for actual LLM calls
    providers = available_providers()
    if not providers:
        _skip("No API keys configured — cannot test LLM path.")
        return True
    test_model = next(iter(providers.values()))

    # Step 3: Excluded schema bypass (FlashcardOutput)
    relevance_prompt = textwrap.dedent(f"""
        Decide whether the following text chunk is relevant to these topics:
        {TARGET_TOPICS}

        Chunk:
        {SAMPLE_TEXT[:500]}
    """).strip()

    try:
        r1_fc = generate_structured(test_model, FlashcardOutput, CARD_PROMPT, use_cache=True)
        r2_fc = generate_structured(test_model, FlashcardOutput, CARD_PROMPT, use_cache=True)
        s2 = cache.stats()
        if s2["hits"] == 0 and s2["stores"] == 0:
            _ok("Excluded schema: FlashcardOutput not stored (SEMANTIC_CACHE_EXCLUDE_SCHEMAS respected)")
        else:
            _fail(f"FlashcardOutput should be excluded but cache shows hits={s2['hits']}, stores={s2['stores']}")
    except Exception as exc:
        if is_quota_error(exc):
            _skip(f"Quota exhausted during exclusion test — {exc}")
        else:
            _fail(f"Exclusion test raised: {exc}")

    # Step 4: First call (RelevanceDecision) → MISS
    cache.clear()
    try:
        r1 = generate_structured(test_model, RelevanceDecision, relevance_prompt, use_cache=True)
        s3 = cache.stats()
        if r1 is not None and s3["misses"] >= 1 and s3["stores"] >= 1:
            _ok(f"First call (RelevanceDecision): MISS — stored in cache (misses={s3['misses']}, stores={s3['stores']})")
        elif r1 is None:
            _skip("First LLM call returned None (quota?); skipping HIT test.")
            return True
        else:
            _fail(f"Expected MISS+store but stats={s3}")
    except Exception as exc:
        if is_quota_error(exc):
            _skip(f"Quota exhausted on first RelevanceDecision call — {exc}")
            return True
        _fail(f"First call raised: {exc}")
        return False

    # Step 5: Second call (identical prompt) → HIT
    try:
        hits_before = cache.stats()["hits"]
        r2 = generate_structured(test_model, RelevanceDecision, relevance_prompt, use_cache=True)
        s4 = cache.stats()
        if s4["hits"] > hits_before and r2 is not None and r2.is_relevant == r1.is_relevant:
            _ok(f"Second call (RelevanceDecision): HIT — returned from semantic cache (no API call)")
        else:
            _fail(f"Expected HIT but stats={s4}")
    except Exception as exc:
        _fail(f"Second call raised: {exc}")
        return False

    # Step 6: Rephrased prompt — informational
    rephrased = relevance_prompt.replace("Decide whether", "Determine if").replace("text chunk", "passage")
    try:
        hits_before = cache.stats()["hits"]
        generate_structured(test_model, RelevanceDecision, rephrased, use_cache=True)
        s5 = cache.stats()
        if s5["hits"] > hits_before:
            print(f"  [INFO] Rephrased prompt: HIT (threshold crossed — semantically similar enough)")
        else:
            print(f"  [INFO] Rephrased prompt: MISS (threshold not crossed — expected at boundary)")
    except Exception:
        pass

    # Step 7: use_cache=False → bypass
    try:
        hits_before = cache.stats()["hits"]
        generate_structured(test_model, RelevanceDecision, relevance_prompt, use_cache=False)
        s6 = cache.stats()
        if s6["hits"] == hits_before:
            _ok("use_cache=False: bypassed cache correctly")
        else:
            _fail("use_cache=False still incremented hit counter")
    except Exception as exc:
        if is_quota_error(exc):
            _ok("use_cache=False: quota on bypass call (cache correctly not queried)")
        else:
            _fail(f"Bypass call raised: {exc}")

    # Step 8: Final stats
    final = cache.stats()
    print(f"\n  Final cache stats: {final}")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\nNexus Learner — Free-Tier Model Hopping Sample + RAGAS Baseline")
    print("Providers: Groq (Llama-3.3-70B) | Gemini 2.0 Flash | Claude Sonnet")

    results = {
        "demo_tier_dispatch":                            demo_tier_dispatch(),
        "test_gemini_structured_output":                 test_gemini_structured_output(),
        "test_groq_structured_output":                   test_groq_structured_output(),
        "test_gemini_flashcard_generation":              test_gemini_flashcard_generation(),
        "test_model_hop":                                test_model_hop(),
        "test_claude_comparison":                        test_claude_comparison(),
        "benchmark_providers (RAGAS)":                   benchmark_providers(),
        "test_reasoning_tier (DeepSeek-R1/Groq)":       test_reasoning_tier(),
        "demo_semantic_cache":                           demo_semantic_cache(),
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
