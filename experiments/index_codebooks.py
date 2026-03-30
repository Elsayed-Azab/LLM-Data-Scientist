"""CLI script to index codebook PDFs into the ChromaDB vector store.

Usage:
    python experiments/index_codebooks.py
    python experiments/index_codebooks.py --pdf Data/GSS_stata/GSS\ 2024\ Codebook\ R2.pdf --dataset gss
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.indexer import CodebookIndexer


def main():
    parser = argparse.ArgumentParser(description="Index codebook PDFs for RAG retrieval")
    parser.add_argument("--pdf", help="Path to a specific codebook PDF")
    parser.add_argument("--dataset", help="Dataset name for the PDF (required with --pdf)")
    parser.add_argument("--persist-dir", default="chroma_db", help="ChromaDB directory")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    parser.add_argument("--stats", action="store_true", help="Show collection stats and exit")
    args = parser.parse_args()

    indexer = CodebookIndexer(
        persist_dir=args.persist_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    if args.stats:
        stats = indexer.get_collection_stats()
        print(f"Collection stats: {stats}")
        return

    if args.pdf:
        if not args.dataset:
            parser.error("--dataset is required when using --pdf")
        count = indexer.index_pdf(args.pdf, dataset_name=args.dataset)
        print(f"Indexed {count} chunks from {args.pdf}")
    else:
        print("Indexing all known codebooks...")
        results = indexer.index_all_codebooks()
        total = sum(results.values())
        print(f"\nDone. Indexed {total} total chunks.")


if __name__ == "__main__":
    main()
