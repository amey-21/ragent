# eval/evaluate.py

"""
LLM-AS-JUDGE EVALUATOR
-----------------------
Instead of Ragas, we use Groq directly to score each response.
This is what many production teams do — Ragas is just a wrapper
around the same idea.

We measure 3 metrics, same as Ragas:

FAITHFULNESS   — are all claims in the answer supported by the context?
                 Score 0.0-1.0. Low = hallucination.

ANSWER QUALITY — does the answer actually address the question?
                 Score 0.0-1.0. Low = vague or off-topic.

CONTEXT RECALL — does the retrieved context contain the information
                 needed to answer the question?
                 Score 0.0-1.0. Low = retrieval is missing relevant docs.

INTERVIEW ANSWER:
"I implemented LLM-as-judge evaluation from scratch using direct Groq
calls. For each test question I ran the full pipeline, then prompted
the LLM to score faithfulness, answer quality, and context recall on
a 0-1 scale with reasoning. This gave me interpretable per-question
diagnostics rather than opaque library scores."
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from groq import Groq
from dotenv import load_dotenv
from src.retrieval.reranker import rerank
from src.agent.graph import SupportSession

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ── LLM judge prompts ──────────────────────────────────────────────────────

def score_faithfulness(question: str, answer: str, context: str) -> dict:
    """
    Ask the LLM: is every claim in the answer supported by the context?
    Returns {"score": 0.0-1.0, "reasoning": "..."}
    """
    prompt = f"""You are an evaluation judge. Score the FAITHFULNESS of an answer.

FAITHFULNESS measures: are all claims in the answer grounded in the provided context?
- Score 1.0: every claim in the answer is directly supported by the context
- Score 0.5: some claims are supported, some are not
- Score 0.0: the answer contains claims not present in the context (hallucination)

QUESTION: {question}

CONTEXT (what was retrieved):
{context}

ANSWER (what the agent said):
{answer}

Respond ONLY with valid JSON, no other text:
{{"score": <float 0.0-1.0>, "reasoning": "<one sentence explanation>"}}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=150,
    )

    raw = response.choices[0].message.content.strip()
    try:
        # strip markdown fences if present
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception:
        return {"score": None, "reasoning": f"Parse error: {raw[:100]}"}


def score_answer_quality(question: str, answer: str) -> dict:
    """
    Ask the LLM: does the answer actually address the question?
    """
    prompt = f"""You are an evaluation judge. Score the ANSWER QUALITY.

ANSWER QUALITY measures: does the answer directly and completely address the question?
- Score 1.0: answer directly addresses the question with useful, specific information
- Score 0.5: answer is partially relevant but vague or incomplete
- Score 0.0: answer does not address the question at all

QUESTION: {question}

ANSWER: {answer}

Respond ONLY with valid JSON, no other text:
{{"score": <float 0.0-1.0>, "reasoning": "<one sentence explanation>"}}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=150,
    )

    raw = response.choices[0].message.content.strip()
    try:
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception:
        return {"score": None, "reasoning": f"Parse error: {raw[:100]}"}


def score_context_recall(question: str, ground_truth: str, context: str) -> dict:
    """
    Ask the LLM: does the retrieved context contain what's needed to answer?
    """
    prompt = f"""You are an evaluation judge. Score the CONTEXT RECALL.

CONTEXT RECALL measures: does the retrieved context contain the information
needed to produce the ground truth answer?
- Score 1.0: context contains all information needed to answer correctly
- Score 0.5: context contains some relevant information but is incomplete
- Score 0.0: context does not contain the information needed to answer

QUESTION: {question}

RETRIEVED CONTEXT:
{context}

GROUND TRUTH ANSWER:
{ground_truth}

Respond ONLY with valid JSON, no other text:
{{"score": <float 0.0-1.0>, "reasoning": "<one sentence explanation>"}}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=150,
    )

    raw = response.choices[0].message.content.strip()
    try:
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception:
        return {"score": None, "reasoning": f"Parse error: {raw[:100]}"}


# ── Main evaluation loop ───────────────────────────────────────────────────

def run_evaluation(dataset_path: str = "eval/dataset.json"):
    with open(dataset_path) as f:
        test_cases = json.load(f)

    print("=" * 65)
    print("LLM-AS-JUDGE EVALUATION")
    print("=" * 65)
    print(f"Questions : {len(test_cases)}")
    print(f"Model     : llama-3.3-70b-versatile (Groq)")
    print(f"Metrics   : faithfulness, answer_quality, context_recall\n")

    rows = []

    for i, case in enumerate(test_cases):
        question     = case["question"]
        ground_truth = case["ground_truth"]

        print(f"[{i+1}/{len(test_cases)}] {question}")

        session = SupportSession(session_id=f"eval_{i}")

        # Get retrieved context
        try:
            retrieved     = rerank(question, top_k=5)
            context_texts = [r["text"] for r in retrieved]
            context_str   = "\n\n---\n\n".join(context_texts)
        except Exception as e:
            print(f"  Retrieval error: {e}")
            context_str = "No context retrieved"

        # Get agent answer
        try:
            result = session.chat(question)
            answer = result["response"]
            tools  = result["tool_calls"]
        except Exception as e:
            print(f"  Agent error: {e}")
            answer = "Error"
            tools  = []

        # Score with LLM judge
        faith   = score_faithfulness(question, answer, context_str)
        time.sleep(1)   # avoid rate limits between judge calls
        quality = score_answer_quality(question, answer)
        time.sleep(1)
        recall  = score_context_recall(question, ground_truth, context_str)
        time.sleep(2)   # longer pause between questions

        row = {
            "question":         question,
            "answer":           answer[:120],
            "tools":            ", ".join(tools),
            "faithfulness":     faith.get("score"),
            "faith_reason":     faith.get("reasoning", ""),
            "answer_quality":   quality.get("score"),
            "quality_reason":   quality.get("reasoning", ""),
            "context_recall":   recall.get("score"),
            "recall_reason":    recall.get("reasoning", ""),
        }
        rows.append(row)

        print(f"  Tools      : {tools}")
        print(f"  Faithful   : {faith.get('score')}  — {faith.get('reasoning','')[:80]}")
        print(f"  Quality    : {quality.get('score')}  — {quality.get('reasoning','')[:80]}")
        print(f"  Recall     : {recall.get('score')}  — {recall.get('reasoning','')[:80]}")

    # Aggregate scores
    print("\n" + "=" * 65)
    print("AGGREGATE SCORES")
    print("=" * 65)

    for metric in ["faithfulness", "answer_quality", "context_recall"]:
        valid  = [r[metric] for r in rows if r[metric] is not None]
        nans   = len(rows) - len(valid)
        avg    = sum(valid) / len(valid) if valid else 0
        if avg >= 0.8:   verdict = "✓ Good"
        elif avg >= 0.6: verdict = "~ Acceptable"
        else:            verdict = "✗ Needs improvement"
        print(f"  {metric:<22} {avg:.4f}   {verdict}   (scored {len(valid)}/{len(rows)})")

    # Save results
    os.makedirs("eval", exist_ok=True)
    with open("eval/results.json", "w") as f:
        json.dump(rows, f, indent=2)

    # Save CSV
    import csv
    with open("eval/results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDetailed results → eval/results.json")
    print(f"CSV results      → eval/results.csv")

    return rows


if __name__ == "__main__":
    run_evaluation()