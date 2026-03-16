"""
scripts/test_haiku_structured_output.py
-----------------------------------------
Standalone diagnostic for the Haiku 4.5 structured-output failure via LiteLLM.

Background
----------
During Phase 2.6 development, `anthropic/claude-haiku-4-5-20251001` returned empty
responses when called via `ChatLiteLLM.with_structured_output(schema)`.  The fast tier
was temporarily switched to `claude-sonnet-4-6` as a workaround.  This script isolates
the root cause so the cheap Haiku model can be restored to the fast tier.

See scripts/haiku_structured_output_investigation.md for the full hypothesis list.

Run
---
    PYTHONPATH=. python scripts/test_haiku_structured_output.py

Requirements
------------
    ANTHROPIC_API_KEY must be set (real paid-tier key).
    pip install langchain-litellm litellm anthropic
"""

from __future__ import annotations

import os
import sys
import textwrap
import traceback
from typing import Optional

# ---------------------------------------------------------------------------
# Minimal Pydantic schema for structured output tests
# ---------------------------------------------------------------------------
from pydantic import BaseModel, Field


class TopicLabel(BaseModel):
    topic: str = Field(description="A broad topic category, e.g. 'Probability Theory'.")
    subtopic: str = Field(description="A specific subtopic, e.g. 'Conditional Probability'.")
    reasoning: str = Field(description="One sentence explaining the classification.")


SAMPLE_CHUNK = textwrap.dedent("""
    Definition (Conditional Probability):
    Let A and B be events with P(B) > 0. The conditional probability of A given B is:
        P(A | B) = P(A ∩ B) / P(B)
    This represents how the probability of A changes when we know B has occurred.
""").strip()

MODELS = {
    "haiku_4_5":  "anthropic/claude-haiku-4-5-20251001",
    "sonnet_4_6": "anthropic/claude-sonnet-4-6",        # known-good control
}

PASS = "✅ PASS"
FAIL = "❌ FAIL"
SKIP = "⏭  SKIP"


def _header(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _result(label: str, status: str, detail: str = "") -> None:
    print(f"  {status}  {label}")
    if detail:
        for line in detail.strip().splitlines():
            print(f"         {line}")


# ---------------------------------------------------------------------------
# Test 1 — Plain completion (no structured output)
# Confirms the model is reachable and returns non-empty text.
# ---------------------------------------------------------------------------

def test_plain_completion(model_key: str) -> bool:
    _header(f"Test 1 — Plain completion [{model_key}]")
    model = MODELS[model_key]
    try:
        from langchain_litellm import ChatLiteLLM
        llm = ChatLiteLLM(model=model, temperature=0)
        result = llm.invoke("Say exactly: HELLO")
        content = getattr(result, "content", "") or ""
        if content.strip():
            _result("Model responds with non-empty text", PASS, f"Response: {content[:80]!r}")
            return True
        else:
            _result("Model responded but content is empty", FAIL)
            return False
    except Exception as e:
        _result("API call failed", FAIL, str(e)[:200])
        return False


# ---------------------------------------------------------------------------
# Test 2 — Structured output via tool calling (function_calling)
# LiteLLM maps this to Anthropic's tools API.
# ---------------------------------------------------------------------------

def test_structured_function_calling(model_key: str) -> bool:
    _header(f"Test 2 — Structured output: method=function_calling [{model_key}]")
    model = MODELS[model_key]
    try:
        from langchain_litellm import ChatLiteLLM
        llm = ChatLiteLLM(model=model, temperature=0)
        chain = llm.with_structured_output(TopicLabel, method="function_calling")
        result = chain.invoke(f"Classify this text:\n\n{SAMPLE_CHUNK}")
        if result and result.topic:
            _result("Structured output returned valid TopicLabel", PASS,
                    f"topic={result.topic!r}, subtopic={result.subtopic!r}")
            return True
        else:
            _result("Structured output returned None or empty fields", FAIL,
                    f"result={result!r}")
            return False
    except Exception as e:
        _result("method=function_calling failed", FAIL,
                f"{type(e).__name__}: {str(e)[:200]}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Test 3 — Structured output via JSON mode
# LiteLLM sends a system prompt instructing the model to respond in JSON.
# ---------------------------------------------------------------------------

def test_structured_json_mode(model_key: str) -> bool:
    _header(f"Test 3 — Structured output: method=json_mode [{model_key}]")
    model = MODELS[model_key]
    try:
        from langchain_litellm import ChatLiteLLM
        llm = ChatLiteLLM(model=model, temperature=0)
        chain = llm.with_structured_output(TopicLabel, method="json_mode")
        result = chain.invoke(f"Classify this text:\n\n{SAMPLE_CHUNK}")
        if result and result.topic:
            _result("JSON mode returned valid TopicLabel", PASS,
                    f"topic={result.topic!r}, subtopic={result.subtopic!r}")
            return True
        else:
            _result("JSON mode returned None or empty fields", FAIL,
                    f"result={result!r}")
            return False
    except Exception as e:
        _result("method=json_mode failed", FAIL,
                f"{type(e).__name__}: {str(e)[:200]}")
        return False


# ---------------------------------------------------------------------------
# Test 4 — LiteLLM model registry check
# Verifies capability flags for the model name.
# ---------------------------------------------------------------------------

def test_litellm_registry(model_key: str) -> None:
    _header(f"Test 4 — LiteLLM model registry [{model_key}]")
    model = MODELS[model_key]
    try:
        import litellm
        params = litellm.get_supported_openai_params(model=model, custom_llm_provider="anthropic")
        tools_supported = params is not None and "tools" in params
        tool_choice_supported = params is not None and "tool_choice" in params
        _result("Model in LiteLLM registry", PASS if params else FAIL,
                f"params={params}")
        _result("tools param supported", PASS if tools_supported else FAIL)
        _result("tool_choice param supported", PASS if tool_choice_supported else FAIL)
    except Exception as e:
        _result("Registry check failed", FAIL, str(e)[:200])


# ---------------------------------------------------------------------------
# Test 5 — Direct Anthropic SDK tool call (bypasses LiteLLM entirely)
# If this works but Test 2 fails → LiteLLM dispatch is the bug.
# If this also fails → Anthropic API level issue (model restriction or quota).
# ---------------------------------------------------------------------------

def test_direct_anthropic_tool_call(model_key: str) -> bool:
    _header(f"Test 5 — Direct Anthropic SDK tool call [{model_key}]")
    model_name = MODELS[model_key].replace("anthropic/", "")
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        _result("ANTHROPIC_API_KEY not set", SKIP)
        return False
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model_name,
            max_tokens=256,
            tools=[{
                "name": "classify_topic",
                "description": "Classify a text chunk into a topic and subtopic.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "topic":     {"type": "string", "description": "Broad topic category"},
                        "subtopic":  {"type": "string", "description": "Specific subtopic"},
                        "reasoning": {"type": "string", "description": "Brief reasoning"},
                    },
                    "required": ["topic", "subtopic", "reasoning"],
                },
            }],
            tool_choice={"type": "auto"},
            messages=[{"role": "user", "content": f"Classify:\n\n{SAMPLE_CHUNK}"}],
        )
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if tool_uses:
            inp = tool_uses[0].input
            _result("Direct tool call returned tool_use block", PASS,
                    f"topic={inp.get('topic')!r}, subtopic={inp.get('subtopic')!r}")
            return True
        else:
            content_types = [b.type for b in response.content]
            _result("No tool_use block in response", FAIL,
                    f"stop_reason={response.stop_reason!r}, content types={content_types}")
            return False
    except Exception as e:
        _result("Direct SDK call failed", FAIL,
                f"{type(e).__name__}: {str(e)[:200]}")
        return False


# ---------------------------------------------------------------------------
# Test 6 — Default with_structured_output (no method override)
# Captures exactly what the app currently does.
# ---------------------------------------------------------------------------

def test_default_structured_output(model_key: str) -> bool:
    _header(f"Test 6 — Default with_structured_output (no method) [{model_key}]")
    model = MODELS[model_key]
    try:
        from langchain_litellm import ChatLiteLLM
        from langchain_core.prompts import ChatPromptTemplate
        llm = ChatLiteLLM(model=model, temperature=0)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an academic topic classifier. Return structured output."),
            ("user", "Classify this text:\n\n{text}"),
        ])
        chain = prompt | llm.with_structured_output(TopicLabel)
        result = chain.invoke({"text": SAMPLE_CHUNK})
        if result and result.topic:
            _result("Default path returned valid TopicLabel", PASS,
                    f"topic={result.topic!r}")
            return True
        else:
            _result("Default path returned None or empty", FAIL, f"result={result!r}")
            return False
    except Exception as e:
        _result("Default path failed", FAIL,
                f"{type(e).__name__}: {str(e)[:200]}")
        return False


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_model(model_key: str) -> dict:
    print(f"\n{'#'*60}")
    print(f"  DIAGNOSING: {MODELS[model_key]}")
    print(f"{'#'*60}")

    reachable   = test_plain_completion(model_key)
    fc_result   = test_structured_function_calling(model_key) if reachable else None
    json_result = test_structured_json_mode(model_key)        if reachable else None
    test_litellm_registry(model_key)
    direct_ok   = test_direct_anthropic_tool_call(model_key)
    default_ok  = test_default_structured_output(model_key)   if reachable else None

    return {
        "model":            model_key,
        "reachable":        reachable,
        "function_calling": fc_result,
        "json_mode":        json_result,
        "direct_api":       direct_ok,
        "default_path":     default_ok,
    }


def print_diagnosis(results: list[dict]) -> None:
    _header("DIAGNOSIS SUMMARY")

    haiku  = next((r for r in results if r["model"] == "haiku_4_5"), None)
    sonnet = next((r for r in results if r["model"] == "sonnet_4_6"), None)

    if haiku is None:
        print("  Haiku not tested.")
        return

    if not haiku["reachable"]:
        print("  ▶ Haiku is NOT reachable — check ANTHROPIC_API_KEY and account tier.")
        return

    if haiku["direct_api"] and not haiku["function_calling"]:
        print("  ▶ H1/H2 CONFIRMED: Anthropic API supports tool calling for Haiku 4.5,")
        print("    but LiteLLM's ChatLiteLLM dispatch is broken for this model name.")
        print("    Fix: upgrade LiteLLM, or use bind_structured() with explicit method='function_calling'.")
    elif not haiku["direct_api"]:
        print("  ▶ H3/H5 CONFIRMED: Anthropic API does not support tool calling for Haiku 4.5")
        print("    (or account is restricted). LiteLLM is not the issue.")
        print("    Fix: use Groq llama-3.3-70b-versatile as fast-tier primary.")
    elif haiku["function_calling"] and not haiku["default_path"]:
        print("  ▶ H2 LIKELY: explicit method=function_calling works but default dispatch fails.")
        print("    Fix: use bind_structured() with method='function_calling' for Anthropic models.")
    elif haiku["json_mode"] and not haiku["function_calling"]:
        print("  ▶ Haiku supports JSON mode but not tool calling via LiteLLM.")
        print("    Fix: use method=json_mode in bind_structured() for Haiku models.")
    elif haiku["default_path"]:
        print("  ▶ Haiku NOW WORKS on the default path — the original bug may have been")
        print("    a transient API issue or LiteLLM version has been updated.")
        print("    Restore claude-haiku-4-5-20251001 as fast tier primary.")
    else:
        print("  ▶ All structured output paths fail. See individual test output above.")

    if sonnet and haiku["reachable"]:
        sonnet_works = all(v for v in [sonnet["function_calling"], sonnet["default_path"]] if v is not None)
        haiku_works  = all(v for v in [haiku["function_calling"],  haiku["default_path"]]  if v is not None)
        if sonnet_works and not haiku_works:
            print("\n  ▶ Model-specific issue confirmed: Sonnet 4.6 works, Haiku 4.5 does not.")


def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key or not api_key.startswith("sk-"):
        print("ERROR: ANTHROPIC_API_KEY is not set or looks like a placeholder.")
        print("       Set it in your shell or .env and re-run.")
        sys.exit(1)

    print("Nexus Learner — Haiku Structured Output Diagnostic")
    print(f"LiteLLM version: ", end="")
    try:
        import litellm; print(litellm.__version__)
    except Exception:
        print("unknown")
    print(f"langchain-litellm: ", end="")
    try:
        import langchain_litellm; print(getattr(langchain_litellm, "__version__", "installed"))
    except Exception:
        print("not installed")

    results = []
    for key in ("haiku_4_5", "sonnet_4_6"):
        results.append(run_model(key))

    print_diagnosis(results)

    # Exit 0 if default path works for Haiku (workaround can be removed)
    haiku = next((r for r in results if r["model"] == "haiku_4_5"), None)
    if haiku and haiku.get("default_path"):
        print("\n  ✅ Haiku default path works — safe to restore as fast-tier primary.\n")
        sys.exit(0)
    else:
        print("\n  ❌ Haiku default path broken — keep Sonnet as fast-tier workaround.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
