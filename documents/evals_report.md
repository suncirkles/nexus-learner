# Demystifying Evals for Nexus Learner Agents

## 1. Executive Summary
As Nexus Learner scales and its AI agents become more complex, relying solely on manual testing and "vibes" to verify their behavior will break down. Implementing a robust, automated evaluation (evals) framework is critical for shipping updates confidently, avoiding regressions, and proving the educational value of the platform. Based on Anthropic's industry-leading practices, this report outlines the current state of Nexus Learner's agents, where to start building evals, how to construct them, and the expected impact on the development lifecycle.

## 2. Current State of Nexus Learner Agents
Nexus Learner utilizes a modular, multi-agent architecture. Key agents include:
*   **SocraticAgent (`socratic.py`)**: Responsible for generating flashcards (Active Recall, Fill-in-the-Blank, etc.) with strict rubrics and complexity levels. This is highly generative and complex.
*   **WebResearchAgent (`web_researcher.py`)**: Handles web scraping, searching trusted domains, and filtering content.
*   **SafetyAgent / Critic / Curator / Ingestion**: Various utility and processing agents that handle pipeline steps.

Currently, these agents rely on complex, carefully crafted prompts (e.g., the `_BASE_RULES` in `socratic.py`) to govern behavior. However, there is no evidence of an automated, multi-turn evaluation harness to systematically verify that these prompts consistently produce high-quality, hallucination-free educational content across diverse subject matter.

## 3. Why Implement Evals for Nexus Learner?
*   **Preventing Regressions ("Flying Blind"):** When you tweak a prompt to fix a bug in numerical flashcards, how do you know you didn't break scenario-based flashcards? Evals catch these regressions automatically.
*   **Hill-Climbing Capabilities:** Evals give you a baseline metric to optimize against when iterating on prompts or upgrading underlying LLMs (e.g., migrating to a newer model).
*   **Defining Success:** Evals force the engineering and product teams to unambiguously define what a "good" flashcard or a "safe" web snippet looks like.

## 4. Where to Implement Evals First
Do not try to evaluate everything at once. Start where the complexity and user impact are highest.

1.  **SocraticAgent (Highest Priority)**
    *   *Why:* This agent directly generates the core educational content (flashcards). Bad flashcards ruin the user experience.
    *   *What to evaluate:* Groundedness (no hallucinations), rubric accuracy (are there exactly 3 criteria?), adherence to question types, and appropriate complexity categorization.
2.  **WebResearchAgent**
    *   *Why:* Hallucinations or retrieving unsafe/irrelevant content poisons the context window for downstream agents.
    *   *What to evaluate:* Relevance of retrieved snippets to the query, successful filtering of untrusted domains, and appropriate triggering of the SafetyAgent.

## 5. How to Implement Evals: The 0-to-1 Roadmap
Following Anthropic's recommended approach:

1.  **Collect an Initial Dataset (Start Small):**
    *   Gather 20-50 real-world examples (e.g., source texts and expected flashcards).
    *   Source these from known failures, edge cases, or manually verified "golden" examples.
2.  **Write Unambiguous Tasks:**
    *   A task for the SocraticAgent would be: "Given this chunk of text about photosynthesis, generate one Active Recall flashcard. The answer must cite the light-dependent reactions."
    *   Create a *reference solution* for each task.
3.  **Build a Stable Eval Harness:**
    *   Create a script that takes the dataset, runs the agent offline (mocking DB calls where necessary, which `socratic.py` already supports nicely via `FlashcardDraft`), and records the transcript.
    *   Ensure the environment is deterministic (e.g., fixed temperature, mocked web responses for the WebResearchAgent).
4.  **Design the Graders:**
    *   **Code-based Graders:** Use simple Python assertions for objective metrics. (e.g., `assert len(flashcard.rubric) == 3`, `assert flashcard.question_type in valid_types`).
    *   **LLM-as-a-Judge (Model-based Graders):** For nuanced evaluation, use an LLM to grade the output. (e.g., "Prompt an LLM to rate if the generated question is fully answerable using *only* the provided source text.")
    *   **Human Graders:** Periodically sample subset of results to calibrate the LLM judge.

## 6. Graders and Metrics
*   **Capability Evals:** Test edge cases (e.g., deeply complex numerical derivations). Expect a lower pass rate initially. Use this to improve the agent.
*   **Regression Evals:** Test standard, easy concepts. These should have a ~100% pass rate. Run on every PR to ensure core functionality doesn't break.

## 7. Impact
Implementing an eval suite will initially require dedicated engineering time (approx. 1-2 weeks for a V1 harness and dataset). However, the compounding ROI is massive:
*   **Faster Iteration:** Developers will confidently merge PRs knowing they haven't broken the Socratic pipeline.
*   **Seamless Model Upgrades:** When evaluating a new frontier model, you can run the eval suite to immediately quantify if the new model is better, worse, or cheaper for Nexus Learner's specific use cases.
*   **Higher Quality Content:** The quality floor of generated educational materials will rise significantly, leading to better learner outcomes.

## Next Steps
1.  **Approve priority:** Confirm that `SocraticAgent` is the correct first target.
2.  **Curate Dataset:** Manually select 20 diverse text chunks to serve as the foundational test set.
3.  **Build Harness:** Develop a lightweight Python test runner (`tests/evals/`) to execute the dataset against the agent.
