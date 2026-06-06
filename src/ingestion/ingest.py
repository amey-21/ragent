# src/ingestion/ingest.py

"""
Master ingestion script — runs the full pipeline:
  1. Chunk all files in data/raw/
  2. Embed all chunks
  3. Store in ChromaDB
  4. Print stats

Run this whenever you update your data.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.ingestion.chunker import chunk_all_files
from src.ingestion.embedder import embed_chunks, get_collection_stats

def main():
    print("=" * 50)
    print("INGESTION PIPELINE")
    print("=" * 50)

    # Step 1: Chunk
    print("\n[1/2] Chunking documents...")
    chunks = chunk_all_files("data/raw", strategy="qa_aware")
    print(f"  ✓ {len(chunks)} chunks from {len(set(c.source for c in chunks))} files")

    # Step 2: Embed + store
    print("\n[2/2] Embedding and storing...")
    embed_chunks(chunks)

    # Stats
    print("\n" + "=" * 50)
    print("INGESTION COMPLETE")
    stats = get_collection_stats()
    print(f"  Total chunks in DB : {stats['total_chunks']}")
    print(f"  Sample IDs         : {stats['sample_ids']}")
    print(f"  Sample topics      : {stats['sample_topics']}")
    print("=" * 50)


if __name__ == "__main__":
    main()