# src/ingestion/download_data.py

"""
Dataset: bitext/Bitext-customer-support-llm-chatbot-training-dataset
- ~26,000 real customer support utterances
- 27 intents: get_refund, payment_issue, cancel_order, track_order, etc.
- Each row has: instruction (customer message) + response (agent reply) + intent

We group rows by intent → each intent becomes one .txt file in data/raw/
So "get_refund.txt" contains all Q&A pairs about refunds, etc.
"""

import os
from datasets import load_dataset
from collections import defaultdict

RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)


def download_and_save():
    print("Downloading dataset from HuggingFace...")
    dataset = load_dataset(
        "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        split="train"
    )

    print(f"Total rows: {len(dataset)}")
    print(f"Columns: {dataset.column_names}")
    print(f"\nSample row:")
    row = dataset[0]
    for k, v in row.items():
        print(f"  {k}: {str(v)[:120]}")

    # Group by intent
    topic_groups = defaultdict(list)
    for row in dataset:
        intent = row.get("intent") or row.get("category") or "general"
        question = row.get("instruction") or row.get("input") or ""
        answer = row.get("response") or row.get("output") or ""
        if question and answer:
            topic_groups[intent].append((question.strip(), answer.strip()))

    print(f"\nFound {len(topic_groups)} intents:")
    for intent, pairs in sorted(topic_groups.items()):
        print(f"  {intent}: {len(pairs)} examples")

    # Save each intent as a .txt file
    # We only keep one representative Q&A per unique question to avoid
    # massive duplication (the dataset has many paraphrases per intent)
    saved = 0
    for intent, pairs in topic_groups.items():
        safe_name = intent.lower().replace(" ", "_").replace("/", "_")
        filepath = os.path.join(RAW_DIR, f"{safe_name}.txt")

        # Deduplicate — keep max 30 pairs per intent (enough for good RAG)
        seen = set()
        unique_pairs = []
        for q, a in pairs:
            key = q[:60].lower()    # rough dedup on first 60 chars
            if key not in seen:
                seen.add(key)
                unique_pairs.append((q, a))
            if len(unique_pairs) >= 30:
                break

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"TOPIC: {intent.replace('_', ' ').title()}\n\n")
            for q, a in unique_pairs:
                f.write(f"Q: {q}\n")
                f.write(f"A: {a}\n\n")

        print(f"  ✓ {safe_name}.txt  ({len(unique_pairs)} Q&A pairs)")
        saved += 1

    print(f"\nDone. {saved} files saved to {RAW_DIR}/")
    return saved


if __name__ == "__main__":
    download_and_save()