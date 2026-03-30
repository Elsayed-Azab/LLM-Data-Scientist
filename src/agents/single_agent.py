"""Single-agent architecture: one LLM handles the full analysis pipeline."""

from __future__ import annotations

import time

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent

from src.agents.base import AnalysisResult, BaseAgent
from src.agents.llm_factory import create_llm
from src.agents.tools import get_all_tools
from src.data.registry import get_dataset_info

SYSTEM_PROMPT = """\
You are a expert data analyst specialising in social science survey data.
You have access to tools that let you load datasets, inspect schemas, and
execute Python/pandas code against real survey data.

CRITICAL RULES:
1. ALWAYS apply survey weights when computing statistics. Use the weight
   column indicated in the dataset metadata.
2. Handle missing values explicitly — drop or document them.
3. Use proper statistical methods (weighted means, cross-tabs, etc.).
4. Store your final result in a variable called ``result``.
5. Provide a clear, concise interpretation of your findings.
6. When loading datasets, ALWAYS specify only the columns you need.
   Do NOT load all columns — some datasets have thousands of columns.
7. For large datasets (especially GSS with 6000+ columns), use the
   search_columns tool FIRST to find the right column names before loading.
   Column names are lowercase in GSS (e.g. 'educ', 'happy', 'marital', 'realinc', 'partyid').

Dataset: {dataset_name} — {dataset_description}
Weight column: {weight_column}
"""


class SingleAgent(BaseAgent):
    """ReAct agent: iteratively reasons and calls tools until done."""

    def __init__(self, model: str = "gpt-4o", provider: str | None = None, temperature: float = 0.2):
        llm = create_llm(model=model, provider=provider, temperature=temperature)
        self.agent = create_react_agent(llm, get_all_tools())

    def analyze(self, question: str, dataset_name: str) -> AnalysisResult:
        info = get_dataset_info(dataset_name)
        system = SYSTEM_PROMPT.format(
            dataset_name=info.name,
            dataset_description=info.description,
            weight_column=info.weight_column,
        )

        t0 = time.time()
        response = self.agent.invoke({
            "messages": [
                SystemMessage(content=system),
                HumanMessage(content=question),
            ]
        })
        elapsed = time.time() - t0

        # Extract the final AI message
        ai_messages = [m for m in response["messages"] if m.type == "ai" and m.content]
        answer = ai_messages[-1].content if ai_messages else ""

        # Collect any code that was executed
        tool_calls = [
            m for m in response["messages"]
            if m.type == "ai" and getattr(m, "tool_calls", None)
        ]
        code_blocks = []
        for m in tool_calls:
            for tc in m.tool_calls:
                if tc["name"] == "run_analysis_code":
                    code_blocks.append(tc["args"].get("code", ""))

        return AnalysisResult(
            question=question,
            dataset=dataset_name,
            answer=answer,
            code_executed="\n\n# ---\n\n".join(code_blocks),
            execution_time_seconds=round(elapsed, 2),
        )
