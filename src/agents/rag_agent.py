"""RAG-augmented agent: retrieves codebook context before analysis.

Extends the single-agent approach by injecting relevant codebook
information into the system prompt, so the LLM knows which variables
map to which survey questions, what value labels mean, etc.

For datasets without indexed codebooks, generates schema-based context
from the dataset's column names, dtypes, and sample values as a fallback.
"""

from __future__ import annotations

import time

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent

from src.agents.base import AnalysisResult, BaseAgent
from src.agents.llm_factory import create_llm
from src.agents.tools import get_all_tools
from src.data.loader import DatasetLoader
from src.data.registry import get_dataset_info
from src.rag.indexer import CodebookIndexer
from src.rag.retriever import CodebookRetriever

SYSTEM_PROMPT = """\
You are an expert data analyst specialising in social science survey data.
You have access to tools that let you load datasets, inspect schemas, and
execute Python/pandas code against real survey data.

CRITICAL RULES:
1. ALWAYS apply survey weights when computing statistics. Use the weight
   column indicated in the dataset metadata.
2. Handle missing values explicitly — drop or document them.
3. Use proper statistical methods (weighted means, cross-tabs, etc.).
4. Store your final result in a variable called ``result``.
5. Provide a clear, concise interpretation of your findings.
6. Use the context below to identify correct variable names and understand
   what coded values mean. When loading data, only request the columns you need.
7. For large datasets (especially GSS with 6000+ columns), use the
   search_columns tool to find the right column names before loading.
   Column names are lowercase in GSS (e.g. 'educ', 'happy', 'marital', 'realinc', 'partyid').

Dataset: {dataset_name} — {dataset_description}
Weight column: {weight_column}

{context}
"""


class RAGAgent(BaseAgent):
    """ReAct agent augmented with codebook retrieval.

    Before sending the question to the LLM, relevant sections of the
    codebook are retrieved from ChromaDB and injected into the system
    prompt. For datasets without codebooks, a schema-based fallback
    context is generated from column names and sample values.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        provider: str | None = None,
        temperature: float = 0.2,
        persist_dir: str = "chroma_db",
        top_k: int = 8,
        auto_index: bool = True,
    ):
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.top_k = top_k
        self.persist_dir = persist_dir
        self.auto_index = auto_index

        llm = create_llm(model=model, provider=provider, temperature=temperature)
        self.agent = create_react_agent(llm, get_all_tools())
        self.retriever = CodebookRetriever(persist_dir=persist_dir, top_k=top_k)
        self._loader = DatasetLoader()

    def _ensure_indexed(self) -> None:
        """Index codebooks if the vector store is empty."""
        if self.auto_index and not self.retriever.is_indexed:
            print("RAGAgent: No codebook index found. Indexing codebooks...")
            indexer = CodebookIndexer(persist_dir=self.persist_dir)
            results = indexer.index_all_codebooks()
            total = sum(results.values())
            print(f"RAGAgent: Indexed {total} chunks across {len(results)} codebook(s).")

    def _get_schema_context(self, dataset_name: str) -> str:
        """Generate context from dataset schema when no codebook is available."""
        try:
            schema = self._loader.get_schema(dataset_name, sample_rows=3)
            all_cols = schema["columns"]

            lines = [f"DATASET SCHEMA ({len(all_cols)} columns):"]

            # Show first 100 columns with dtypes
            show = all_cols[:100]
            for col in show:
                dtype = schema["dtypes"].get(col, "?")
                # Show sample values for each column
                sample_vals = []
                for row in schema.get("sample", []):
                    v = row.get(col)
                    if v is not None:
                        sample_vals.append(str(v))
                sample_str = f"  (e.g. {', '.join(sample_vals[:3])})" if sample_vals else ""
                lines.append(f"  {col}: {dtype}{sample_str}")

            if len(all_cols) > 100:
                lines.append(f"  ... and {len(all_cols) - 100} more columns")
                lines.append("  Use get_variable_info tool to inspect specific columns.")

            return "\n".join(lines)
        except Exception:
            return "Schema context unavailable. Use get_dataset_schema tool to inspect columns."

    def _get_context(self, question: str, dataset_name: str) -> tuple[str, str]:
        """Get the best available context for this dataset.

        Returns (context_text, context_source).
        """
        self._ensure_indexed()

        # Try codebook retrieval first
        chunks = self.retriever.retrieve(
            query=question,
            dataset_name=dataset_name,
            top_k=self.top_k,
        )

        if chunks:
            # Codebook context available
            context = self.retriever.retrieve_as_context(
                query=question,
                dataset_name=dataset_name,
                top_k=self.top_k,
            )
            return context, "codebook"

        # Also try without dataset filter (cross-dataset codebook might help)
        chunks_any = self.retriever.retrieve(
            query=question,
            dataset_name=None,
            top_k=self.top_k,
        )

        if chunks_any:
            context = self.retriever.retrieve_as_context(
                query=question,
                dataset_name=None,
                top_k=self.top_k,
            )
            return context, "codebook_cross"

        # Fallback: generate schema-based context
        context = self._get_schema_context(dataset_name)
        return context, "schema_fallback"

    def analyze(self, question: str, dataset_name: str) -> AnalysisResult:
        info = get_dataset_info(dataset_name)

        context, context_source = self._get_context(question, dataset_name)

        system = SYSTEM_PROMPT.format(
            dataset_name=info.name,
            dataset_description=info.description,
            weight_column=info.weight_column,
            context=context,
        )

        t0 = time.time()
        try:
            response = self.agent.invoke({
                "messages": [
                    SystemMessage(content=system),
                    HumanMessage(content=question),
                ]
            })
        except Exception as e:
            return AnalysisResult(
                question=question,
                dataset=dataset_name,
                answer="",
                execution_time_seconds=round(time.time() - t0, 2),
                errors=[str(e)],
            )
        elapsed = time.time() - t0

        # Extract the final AI message
        ai_messages = [m for m in response["messages"] if m.type == "ai" and m.content]
        answer = ai_messages[-1].content if ai_messages else ""

        # Collect any code that was executed
        code_blocks = []
        for m in response["messages"]:
            if m.type == "ai" and getattr(m, "tool_calls", None):
                for tc in m.tool_calls:
                    if tc["name"] == "run_analysis_code":
                        code_blocks.append(tc["args"].get("code", ""))

        return AnalysisResult(
            question=question,
            dataset=dataset_name,
            answer=answer,
            code_executed="\n\n# ---\n\n".join(code_blocks),
            raw_statistics={
                "context_source": context_source,
                "context_preview": context[:500],
            },
            execution_time_seconds=round(elapsed, 2),
        )
