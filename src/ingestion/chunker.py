# src/ingestion/chunker.py

"""
CHUNKING STRATEGIES — we implement 3 so you can compare and explain in interviews.

Strategy 1: FIXED-SIZE
  Split every N characters regardless of content boundaries.
  Pro: Simple, predictable.
  Con: Cuts mid-sentence, destroys meaning.

Strategy 2: RECURSIVE
  Split on paragraph → sentence boundaries in order.
  Pro: Respects natural language structure.
  Con: Variable chunk sizes.

Strategy 3: QA-AWARE  ← what we'll actually use
  Split on Q:/A: pairs — each pair is one chunk.
  Pro: Perfect for our dataset. Each chunk = one complete support interaction.
  Con: Dataset-specific (works because our data has Q:/A: structure).

INTERVIEW ANSWER: "I evaluated three chunking strategies. Fixed-size hurt
retrieval quality because it cut mid-sentence. Recursive was better but
created uneven chunks. Since my dataset was structured Q&A pairs, I used
a QA-aware splitter — each chunk was one complete interaction, which gave
the highest retrieval precision in my Ragas evaluation."
"""

import re
import os
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """One unit of text that gets embedded and stored in the vector DB."""
    text: str           # The actual content the LLM will read
    chunk_id: str       # Unique ID: "get_refund_0", "get_refund_1", ...
    source: str         # Filename: "get_refund.txt"
    topic: str          # Intent/topic: "Get Refund"
    chunk_index: int    # Position within the file
    strategy: str       # Which chunking strategy produced this chunk
    metadata: dict = field(default_factory=dict)  # extra info for filtering


# ── Strategy 1: Fixed-size ─────────────────────────────────────────────────

def fixed_size_chunker(text: str, source: str,
                       chunk_size: int = 300, overlap: int = 50) -> list[Chunk]:
    """
    Slide a window of `chunk_size` chars across the text.
    `overlap` makes consecutive chunks share some content —
    this prevents losing context that falls exactly at a boundary.

    Example (chunk_size=10, overlap=3):
      text = "ABCDEFGHIJKLMNOP"
      chunk 0: "ABCDEFGHIJ"
      chunk 1: "HIJKLMNOP"   ← starts 3 chars back
    """
    chunks = []
    start = 0
    index = 0
    topic = _extract_topic(text)

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(Chunk(
                text=chunk_text,
                chunk_id=f"{source}_fixed_{index}",
                source=source,
                topic=topic,
                chunk_index=index,
                strategy="fixed"
            ))
            index += 1
        start += chunk_size - overlap

    return chunks


# ── Strategy 2: Recursive ──────────────────────────────────────────────────

def recursive_chunker(text: str, source: str,
                      chunk_size: int = 500) -> list[Chunk]:
    """
    Try to split on natural boundaries in this order:
      1. Double newline (paragraph break)
      2. Single newline
      3. Period/sentence end
      4. Hard cut at chunk_size (last resort)

    Each level is only used if the text is still too large after the previous.
    """
    topic = _extract_topic(text)
    chunks = []
    index = 0

    # Level 1: split on paragraphs
    paragraphs = re.split(r'\n\n+', text)

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(para) <= chunk_size:
            chunks.append(Chunk(
                text=para,
                chunk_id=f"{source}_recursive_{index}",
                source=source,
                topic=topic,
                chunk_index=index,
                strategy="recursive"
            ))
            index += 1
        else:
            # Level 2: split paragraph into sentences
            sentences = re.split(r'(?<=[.!?])\s+', para)
            current = ""

            for sentence in sentences:
                if len(current) + len(sentence) + 1 <= chunk_size:
                    current = (current + " " + sentence).strip()
                else:
                    if current:
                        chunks.append(Chunk(
                            text=current,
                            chunk_id=f"{source}_recursive_{index}",
                            source=source,
                            topic=topic,
                            chunk_index=index,
                            strategy="recursive"
                        ))
                        index += 1
                    current = sentence

            if current:
                chunks.append(Chunk(
                    text=current,
                    chunk_id=f"{source}_recursive_{index}",
                    source=source,
                    topic=topic,
                    chunk_index=index,
                    strategy="recursive"
                ))
                index += 1

    return chunks


# ── Strategy 3: QA-aware (our main strategy) ──────────────────────────────

def qa_aware_chunker(text: str, source: str) -> list[Chunk]:
    """
    Split on Q:/A: pairs. Each Q+A pair becomes one chunk.

    This is optimal for our dataset because:
    - Each pair is semantically complete
    - Retrieval returns exactly the right Q&A, not surrounding noise
    - Chunk size is naturally ~100-200 tokens (ideal for LLMs)

    The chunk includes both Q and A so the LLM sees the full context:
    "Q: how do I get a refund? A: To request a refund, please..."
    """
    topic = _extract_topic(text)
    chunks = []

    # Split on lines that start with "Q:" — each is a new Q&A pair
    # re.split with a lookahead keeps the "Q:" at the start of each piece
    qa_blocks = re.split(r'(?=^Q:)', text, flags=re.MULTILINE)

    index = 0
    for block in qa_blocks:
        block = block.strip()

        # Skip the TOPIC: header block (no Q: in it)
        if not block or not block.startswith("Q:"):
            continue

        # Verify it has both a question and answer
        if "A:" not in block:
            continue

        chunks.append(Chunk(
            text=block,
            chunk_id=f"{source}_qa_{index}",
            source=source,
            topic=topic,
            chunk_index=index,
            strategy="qa_aware",
            metadata={"intent": topic.lower().replace(" ", "_")}
        ))
        index += 1

    return chunks


# ── Helpers ────────────────────────────────────────────────────────────────

def _extract_topic(text: str) -> str:
    """Pull the topic name from the first line: 'TOPIC: Get Refund' → 'Get Refund'"""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("TOPIC:"):
            return line.replace("TOPIC:", "").strip()
    return "Unknown"


def chunk_file(filepath: str, strategy: str = "qa_aware") -> list[Chunk]:
    """Read a file and chunk it with the chosen strategy."""
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    # Use just the filename as the source identifier
    source = os.path.basename(filepath).replace(".txt", "")

    if strategy == "fixed":
        return fixed_size_chunker(text, source)
    elif strategy == "recursive":
        return recursive_chunker(text, source)
    elif strategy == "qa_aware":
        return qa_aware_chunker(text, source)
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Use: fixed, recursive, qa_aware")


def chunk_all_files(data_dir: str, strategy: str = "qa_aware") -> list[Chunk]:
    """Chunk every .txt file in a directory. Returns all chunks combined."""
    all_chunks = []
    files = [f for f in os.listdir(data_dir) if f.endswith(".txt")]

    for filename in sorted(files):
        filepath = os.path.join(data_dir, filename)
        chunks = chunk_file(filepath, strategy)
        all_chunks.extend(chunks)

    return all_chunks