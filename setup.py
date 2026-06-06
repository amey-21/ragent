"""
Run this once after cloning to verify your environment is set up correctly.
Usage: python setup.py
"""

import subprocess
import sys
import os


def check(label, fn):
    try:
        fn()
        print(f"  ✓ {label}")
        return True
    except Exception as e:
        print(f"  ✗ {label} — {e}")
        return False


def main():
    print("=" * 50)
    print("SUPPORT AGENT — ENVIRONMENT CHECK")
    print("=" * 50)

    all_ok = True

    # Python version
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
    else:
        print(f"  ✗ Python {version.major}.{version.minor} — needs 3.10+")
        all_ok = False

    # Package imports
    print("\nPackages:")
    packages = [
        ("groq",                    lambda: __import__("groq")),
        ("langchain",               lambda: __import__("langchain")),
        ("langchain_groq",          lambda: __import__("langchain_groq")),
        ("langgraph",               lambda: __import__("langgraph")),
        ("chromadb",                lambda: __import__("chromadb")),
        ("sentence_transformers",   lambda: __import__("sentence_transformers")),
        ("rank_bm25",               lambda: __import__("rank_bm25")),
        ("fastapi",                 lambda: __import__("fastapi")),
        ("datasets",                lambda: __import__("datasets")),
        ("dotenv",                  lambda: __import__("dotenv")),
    ]

    for name, fn in packages:
        if not check(name, fn):
            all_ok = False

    # .env file
    print("\nEnvironment:")
    if os.path.exists(".env"):
        from dotenv import load_dotenv
        load_dotenv()
        key = os.getenv("GROQ_API_KEY", "")
        if key and key != "your_groq_api_key_here":
            print("  ✓ GROQ_API_KEY loaded")
        else:
            print("  ✗ GROQ_API_KEY not set — edit .env")
            all_ok = False
    else:
        print("  ✗ .env not found — copy .env.example to .env")
        all_ok = False

    # Data directory
    print("\nData:")
    raw_files = []
    if os.path.exists("data/raw"):
        raw_files = [f for f in os.listdir("data/raw") if f.endswith(".txt")]
    if raw_files:
        print(f"  ✓ data/raw/ — {len(raw_files)} files found")
    else:
        print("  ! data/raw/ empty — run: python src/ingestion/download_data.py")

    if os.path.exists("data/chroma_db"):
        print("  ✓ ChromaDB index exists")
    else:
        print("  ! ChromaDB not built — run: python src/ingestion/ingest.py")

    # Summary
    print("\n" + "=" * 50)
    if all_ok:
        print("✓ Environment ready.")
        if not raw_files:
            print("\nNext steps:")
            print("  1. python src/ingestion/download_data.py")
            print("  2. python src/ingestion/ingest.py")
            print("  3. uvicorn src.api.main:app --reload --port 8000")
    else:
        print("✗ Some checks failed — fix above issues then re-run.")
        print("  Install packages: pip install -r requirements.txt")


if __name__ == "__main__":
    main()
