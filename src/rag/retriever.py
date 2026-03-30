"""Retrieve relevant codebook context from the ChromaDB vector store."""

from __future__ import annotations

from pathlib import Path

import chromadb

from src.rag.indexer import COLLECTION_NAME, DEFAULT_PERSIST_DIR


class CodebookRetriever:
    """Query the codebook vector store for context relevant to a question."""

    def __init__(
        self,
        persist_dir: str | Path = DEFAULT_PERSIST_DIR,
        top_k: int = 5,
    ):
        self.top_k = top_k
        self.client = chromadb.PersistentClient(path=str(persist_dir))

    def retrieve(
        self,
        query: str,
        dataset_name: str | None = None,
        top_k: int | None = None,
    ) -> list[dict]:
        """Retrieve the most relevant codebook chunks for a query.

        Args:
            query: Natural language question or search terms.
            dataset_name: Optional filter to a specific dataset's codebook.
            top_k: Override the default number of results.

        Returns:
            List of dicts with keys: 'text', 'dataset', 'source', 'distance'.
        """
        k = top_k or self.top_k

        try:
            collection = self.client.get_collection(COLLECTION_NAME)
        except Exception:
            return []

        if collection.count() == 0:
            return []

        where_filter = {"dataset": dataset_name} if dataset_name else None

        results = collection.query(
            query_texts=[query],
            n_results=min(k, collection.count()),
            where=where_filter,
        )

        chunks = []
        for i in range(len(results["documents"][0])):
            chunks.append({
                "text": results["documents"][0][i],
                "dataset": results["metadatas"][0][i].get("dataset", ""),
                "source": results["metadatas"][0][i].get("source", ""),
                "distance": results["distances"][0][i] if results.get("distances") else None,
            })
        return chunks

    def retrieve_as_context(
        self,
        query: str,
        dataset_name: str | None = None,
        top_k: int | None = None,
    ) -> str:
        """Retrieve and format codebook context as a single string for LLM injection.

        Args:
            query: Natural language question.
            dataset_name: Optional dataset filter.
            top_k: Number of chunks to retrieve.

        Returns:
            Formatted context string ready for prompt injection.
        """
        chunks = self.retrieve(query, dataset_name=dataset_name, top_k=top_k)

        if not chunks:
            return "No codebook context available."

        parts = ["RELEVANT CODEBOOK CONTEXT:"]
        for i, chunk in enumerate(chunks, 1):
            parts.append(f"\n--- Chunk {i} (from {chunk['source']}) ---")
            parts.append(chunk["text"])

        return "\n".join(parts)

    @property
    def is_indexed(self) -> bool:
        """Check whether the codebook collection has any documents."""
        try:
            collection = self.client.get_collection(COLLECTION_NAME)
            return collection.count() > 0
        except Exception:
            return False
