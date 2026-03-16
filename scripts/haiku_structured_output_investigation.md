# Investigation: Haiku Structured Output Failure via LiteLLM

**Date:** 2026-03-16
**Branch:** feat/phase-2.5-atomic-content-foundation
**Status:** Unresolved — fast tier temporarily using claude-sonnet-4-6 as workaround

---

## Problem Statement

The fast tier (`MODEL_HOP_ROUTING_TIER=fast`) is intended for cheap, low-latency calls:
topic assignment, relevance scoring, safety classification, topic matching.
These are called once per chunk during indexing and once per card during generation —
at scale (100-page PDF, ~200 chunks) that is 200–400 routing calls.

At current Anthropic pricing, Sonnet 4.6 is roughly **5–10x more expensive per token**
than Haiku. Every routing call paying Sonnet price defeats the entire tiered model strategy.

---

## What Was Observed

### Attempt 1: `anthropic/claude-haiku-4-5-20251001` (Haiku 4.5)

**Symptom:** `OutputParserException: Invalid json output: ` (empty string after the colon)

The model was called, did not raise an API exception, but returned an empty response body.
The output parser received `""` and threw `JSONDecodeError: Expecting value: line 1 col 1`.

```
langchain_core.exceptions.OutputParserException: Invalid json output:
For troubleshooting: https://docs.langchain.com/oss/python/langchain/errors/OUTPUT_PARSING_FAILURE
```

Traceback path:
```
agents/topic_assigner.py → call_structured(TopicAssignment, prompt_str, purpose="routing")
core/models.py:124 → llm.with_structured_output(schema).invoke(prompt)
langchain_core/output_parsers/pydantic.py:75 → json_object = super().parse_result(result)
langchain_core/output_parsers/json.py:88 → parse_json_markdown(text)
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

Note: the parser chain is `pydantic.py → json.py`, indicating `with_structured_output`
is using **JSON mode** (not tool calling) for this model via LiteLLM.

### Attempt 2: `anthropic/claude-3-5-haiku-20241022` (Haiku 3.5)

**Symptom:** `404 Not Found` from `api.anthropic.com/v1/messages`

```
litellm.exceptions.NotFoundError: litellm.NotFoundError: AnthropicException -
{"type":"error","error":{"type":"not_found_error","message":"model: claude-3-5-haiku-20241022"}}
```

This model ID is valid on standard Anthropic accounts but was not found on this account.
Possible reasons: API tier restriction, region, or the account is on a newer API version
where the old naming convention is deprecated.

### Control: `anthropic/claude-sonnet-4-6` (Sonnet 4.6) — WORKS

Used for `balanced` and `reasoning` tiers. All structured output calls succeed.
The RAG quality tests and Sheldon Ross integration tests confirm this model works
correctly with `ChatLiteLLM.with_structured_output(schema).invoke(messages)`.

---

## Root Cause Hypotheses

### H1 — LiteLLM JSON mode vs tool calling dispatch

`ChatLiteLLM.with_structured_output(schema)` selects the output method (tool calling
or JSON mode) based on the model name. If LiteLLM's model registry does not recognise
`claude-haiku-4-5-20251001` as supporting tool calling, it falls back to JSON mode.
In JSON mode the model is expected to return raw JSON; if it instead returns an
explanation or nothing, the parser gets an empty string.

**Evidence for:** The parser chain goes through `pydantic.py → json.py` not
through a tool-call parser. Sonnet 4.6 (same provider, works) likely gets routed
through tool calling.

**Test:** Call `llm.with_structured_output(schema, method="function_calling")` explicitly
and `llm.with_structured_output(schema, method="json_mode")` separately to isolate which
path is taken and which one returns a valid response.

### H2 — LiteLLM version does not know the new Anthropic model-name format

LiteLLM maintains an internal model registry (`litellm/models.py`) that maps model names
to their capabilities. Models with the new naming convention (`claude-haiku-4-5-20251001`
vs the old `claude-3-5-haiku-20241022`) may not be in the registry for the installed
LiteLLM version, causing capability detection to fail silently.

**Evidence for:** Same account, same API key — Sonnet 4.6 (new naming) works, Haiku 4.5
(new naming) does not. Both follow the new convention, so this alone doesn't explain it,
but the capability flags may differ.

**Test:** `litellm.get_supported_openai_params("anthropic/claude-haiku-4-5-20251001")` to
see if the model is in the registry and what params are flagged as supported.

### H3 — Haiku 4.5 does not support tool calling via the Anthropic API

Anthropic may have restricted tool calling / function calling to Sonnet-class and above
for the `claude-haiku-4-5` generation. The model exists but silently ignores the
`tools` parameter and returns no content.

**Evidence for:** The empty response (not an error) is consistent with the model receiving
a tool-call request, not knowing what to do with it, and returning an empty `stop_reason`.

**Test:** Direct `anthropic` SDK call (not via LiteLLM) with `tools=[...]` to see if
tool calling is supported. Compare with a plain completion call.

### H4 — `invoke(prompt_str)` path (pre-fix) collapsed system/user roles

Before the `topic_assigner.py` fix, `call_structured` received a flat string from
`prompt.format(...)` and passed it as a single HumanMessage. The system prompt
instructions were embedded in the human turn. Haiku may be more sensitive to this
than Sonnet and simply refuse to produce structured output without proper system/user
separation.

**Evidence against:** The fix (proper chain) was applied and the error persisted with
identical symptoms, ruling this out as the sole cause.

### H5 — Account/API tier access restriction

The account may be on an API tier that allows Haiku 4.5 for plain completions but not
for tool-calling / structured output endpoints.

**Test:** Plain (non-structured) completion call to `claude-haiku-4-5-20251001` to
verify the model responds at all.

---

## Current Workaround

`TIER_MODELS["fast"]` temporarily set to `anthropic/claude-sonnet-4-6` as primary,
with Groq as fallback. This is tracked as **P1-6** in the project backlog.

---

## What Needs to Be Tested (run `scripts/test_haiku_structured_output.py`)

When API limits reset, run the diagnostic script. It tests:

1. Plain completion (no structured output) → confirms model connectivity
2. `with_structured_output(method="function_calling")` → tool-call path
3. `with_structured_output(method="json_mode")` → JSON mode path
4. LiteLLM model registry check → capability flags
5. Direct Anthropic SDK tool call → isolates LiteLLM vs API issue
6. Side-by-side comparison with Sonnet 4.6 → confirms it's model-specific

If tests 1 and 5 pass but 2 and 3 fail: the issue is LiteLLM's dispatch for this model.
If test 5 also fails: the issue is at the Anthropic API level (H3 or H5).
If test 1 fails: account/quota issue unrelated to structured output.

---

## Resolution Path

| Outcome | Fix |
|---|---|
| H1 confirmed (JSON mode selected, use tool calling) | Force `method="function_calling"` in `bind_structured()` for Anthropic models |
| H2 confirmed (model not in LiteLLM registry) | Upgrade LiteLLM or add manual capability override |
| H3 confirmed (Haiku 4.5 has no tool calling) | Use Groq llama-3.3-70b as fast tier primary (free, fast, supports tools) |
| H4 was the cause (now fixed) | Re-run Sheldon Ross test with Haiku 4.5 to verify |
| H5 confirmed (account restriction) | Use Groq as fast tier primary |

---

## Files Changed as Part of Workaround

- `scripts/model_hop.py`: TIER_MODELS fast tier (lines 58–63)
- Remove workaround once root cause is confirmed and fixed
