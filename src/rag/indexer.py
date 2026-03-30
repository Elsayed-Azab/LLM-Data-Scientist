"""Index codebook documents into ChromaDB for retrieval."""

from __future__ import annotations

from pathlib import Path

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.data.codebook import extract_codebook_sections, parse_pdf_pages

# Default settings (overridable via config)
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "codebooks"


class CodebookIndexer:
    """Parse codebook PDFs and store chunks in a ChromaDB collection.

    Uses ChromaDB's built-in all-MiniLM-L6-v2 embeddings (local, no API key).
    """

    def __init__(
        self,
        persist_dir: str | Path = DEFAULT_PERSIST_DIR,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        self.persist_dir = str(persist_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " "],
        )

    def index_pdf(
        self,
        pdf_path: str | Path,
        dataset_name: str,
        use_sections: bool = True,
    ) -> int:
        """Parse a PDF and add its chunks to the collection.

        Args:
            pdf_path: Path to the codebook PDF.
            dataset_name: Name of the dataset (used as metadata for filtering).
            use_sections: If True, try to split by variable sections first,
                          then sub-chunk. If False, split page-level text.

        Returns:
            Number of chunks added.
        """
        pdf_path = Path(pdf_path)
        collection = self.client.get_or_create_collection(COLLECTION_NAME)

        if use_sections:
            raw_sections = extract_codebook_sections(pdf_path)
        else:
            pages = parse_pdf_pages(pdf_path)
            raw_sections = [p["text"] for p in pages]

        # Sub-chunk long sections
        chunks: list[str] = []
        for section in raw_sections:
            if len(section) <= self.chunk_size:
                chunks.append(section)
            else:
                chunks.extend(self.splitter.split_text(section))

        if not chunks:
            return 0

        # Build IDs and metadata
        ids = [f"{dataset_name}_{i}" for i in range(len(chunks))]
        metadatas = [{"dataset": dataset_name, "source": pdf_path.name, "chunk_index": i}
                     for i in range(len(chunks))]

        # Upsert in batches (ChromaDB max batch ~41666)
        batch_size = 5000
        for start in range(0, len(chunks), batch_size):
            end = start + batch_size
            collection.upsert(
                ids=ids[start:end],
                documents=chunks[start:end],
                metadatas=metadatas[start:end],
            )

        return len(chunks)

    def index_all_codebooks(self, base_path: str | Path = ".") -> dict[str, int]:
        """Index all known codebook PDFs.

        Returns:
            Dict mapping dataset name to number of chunks indexed.
        """
        base = Path(base_path)
        results = {}

        codebooks = {
            "gss": base / "Data" / "GSS_stata" / "GSS 2024 Codebook R2.pdf",
            # Add more codebook PDFs here as they become available:
            # "arab_barometer": base / "Data" / "..." / "codebook.pdf",
            # "wvs": base / "Data" / "..." / "codebook.pdf",
        }

        for name, path in codebooks.items():
            if path.exists():
                count = self.index_pdf(path, dataset_name=name)
                results[name] = count
                print(f"  Indexed {name}: {count} chunks from {path.name}")
            else:
                print(f"  Skipped {name}: {path} not found")

        return results

    def get_collection_stats(self) -> dict:
        """Return stats about the indexed collection."""
        try:
            collection = self.client.get_collection(COLLECTION_NAME)
            return {"count": collection.count()}
        except Exception:
            return {"count": 0}
