"""Multi-agent architecture: Planner → Analyst → Reviewer pipeline.

Uses a LangGraph StateGraph where three specialised LLMs collaborate:
  1. Planner  — receives the question + schema, outputs an analysis plan.
  2. Analyst  — receives the plan, writes and executes code, returns raw results.
  3. Reviewer — validates the results, checks for common errors (missing weights,
                wrong variables), and produces the final interpretation.
If the Reviewer finds issues it can send the analysis back for a retry
(up to MAX_RETRIES).
"""

from __future__ import annotations

import time
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from src.agents.llm_factory import create_llm

from src.agents.base import AnalysisResult, BaseAgent
from src.agents.tools import get_all_tools, run_analysis_code, get_dataset_schema, load_dataset, get_variable_info
from src.data.metadata import get_variable_labels
from src.data.registry import get_dataset_info

MAX_RETRIES = 2

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PLANNER_PROMPT = """\
You are a senior survey-data research planner. Given a research question and
a dataset schema, produce a concrete analysis plan.

Your plan MUST include:
1. Which variables (columns) to use and why.
2. How to handle missing values for those variables.
3. Which statistical method(s) to apply (e.g. weighted mean, cross-tab, \
regression).
4. The survey weight column that MUST be used: {weight_column}.
5. Expected output format (table, single number, comparison, etc.).

Be specific — name exact column names from the schema. Do NOT write code.

Dataset: {dataset_name} — {dataset_description}
Weight column: {weight_column}
"""

ANALYST_PROMPT = """\
You are an expert Python data analyst. You will receive an analysis plan
and must implement it by calling tools to load data and execute code.

CRITICAL RULES:
1. ALWAYS apply survey weights using the weight column: {weight_column}.
2. Handle missing values as specified in the plan.
3. Store your final result in a variable called ``result``.
4. Use print() for intermediate diagnostics.
5. Do NOT interpret the results — just produce the numbers/tables.
6. When loading datasets, ALWAYS specify only the columns you need.
   Do NOT load all columns — some datasets have thousands of columns.
7. For large datasets (especially GSS with 6000+ columns), use the
   search_columns tool FIRST to find the right column names before loading.
   Column names are lowercase in GSS (e.g. 'educ', 'happy', 'marital', 'realinc', 'partyid').

Dataset: {dataset_name} — {dataset_description}
Weight column: {weight_column}
"""

REVIEWER_PROMPT = """\
You are a meticulous statistical reviewer for social science survey analysis.
You will receive:
  - The original research question
  - The analysis plan
  - The code that was executed and its output

Your job:
1. Check that survey weights ({weight_column}) were correctly applied.
2. Check that the right variables were used for the question.
3. Check that missing values were handled appropriately.
4. Check that the statistical method matches the plan.
5. If there are problems, respond with EXACTLY the prefix "REVISION NEEDED:"
   followed by a description of what must be fixed.
6. If everything is correct, provide a clear, concise interpretation of the
   findings that directly answers the research question.

Dataset: {dataset_name}
Weight column: {weight_column}
"""


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class MultiAgentState(TypedDict):
    """Shared state flowing through the graph."""
    question: str
    dataset_name: str
    dataset_description: str
    weight_column: str
    schema_info: str
    plan: str
    analyst_messages: list  # full message history for the analyst ReAct loop
    code_executed: str
    analysis_output: str
    review: str
    retry_count: int
    final_answer: str


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def _build_planner(model: str, temperature: float, provider: str | None = None):
    llm = create_llm(model=model, provider=provider, temperature=temperature)

    def plan_node(state: MultiAgentState) -> dict:
        system = PLANNER_PROMPT.format(
            dataset_name=state["dataset_name"],
            dataset_description=state["dataset_description"],
            weight_column=state["weight_column"],
        )
        response = llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=(
                f"Dataset schema:\n{state['schema_info']}\n\n"
                f"Research question: {state['question']}"
            )),
        ])
        return {"plan": response.content}

    return plan_node


def _build_analyst(model: str, temperature: float, provider: str | None = None):
    from langgraph.prebuilt import create_react_agent

    llm = create_llm(model=model, provider=provider, temperature=temperature)
    tools = get_all_tools()
    react_agent = create_react_agent(llm, tools)

    def analyst_node(state: MultiAgentState) -> dict:
        system = ANALYST_PROMPT.format(
            dataset_name=state["dataset_name"],
            dataset_description=state["dataset_description"],
            weight_column=state["weight_column"],
        )

        # Build the human message — include revision feedback if retrying
        human_parts = [f"Analysis plan:\n{state['plan']}"]
        if state.get("review", "").startswith("REVISION NEEDED:"):
            human_parts.append(
                f"\n\nPrevious attempt was rejected by reviewer:\n{state['review']}\n"
                "Please fix the issues and re-run the analysis."
            )
        human_content = "\n".join(human_parts)

        response = react_agent.invoke(
            {"messages": [
                SystemMessage(content=system),
                HumanMessage(content=human_content),
            ]},
            {"recursion_limit": 30},
        )

        messages = response["messages"]

        # Extract code blocks that were executed
        code_blocks = []
        for m in messages:
            if m.type == "ai" and getattr(m, "tool_calls", None):
                for tc in m.tool_calls:
                    if tc["name"] == "run_analysis_code":
                        code_blocks.append(tc["args"].get("code", ""))

        # Extract tool outputs
        tool_outputs = []
        for m in messages:
            if m.type == "tool":
                tool_outputs.append(m.content)

        # Get final AI message
        ai_msgs = [m for m in messages if m.type == "ai" and m.content]
        analyst_summary = ai_msgs[-1].content if ai_msgs else ""

        code_str = "\n\n# ---\n\n".join(code_blocks)
        output_str = "\n---\n".join(tool_outputs[-3:])  # last 3 tool outputs

        return {
            "code_executed": code_str,
            "analysis_output": f"{analyst_summary}\n\nTool outputs:\n{output_str}",
        }

    return analyst_node


def _build_reviewer(model: str, temperature: float, provider: str | None = None):
    llm = create_llm(model=model, provider=provider, temperature=temperature)

    def review_node(state: MultiAgentState) -> dict:
        system = REVIEWER_PROMPT.format(
            dataset_name=state["dataset_name"],
            weight_column=state["weight_column"],
        )
        response = llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=(
                f"Research question: {state['question']}\n\n"
                f"Analysis plan:\n{state['plan']}\n\n"
                f"Code executed:\n```python\n{state['code_executed']}\n```\n\n"
                f"Analysis output:\n{state['analysis_output']}"
            )),
        ])
        review_text = response.content

        updates: dict = {"review": review_text}
        if review_text.startswith("REVISION NEEDED:"):
            updates["retry_count"] = state.get("retry_count", 0) + 1
        else:
            updates["final_answer"] = review_text
        return updates

    return review_node


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def _should_retry(state: MultiAgentState) -> str:
    """Route after review: retry analysis or finish."""
    if (
        state.get("review", "").startswith("REVISION NEEDED:")
        and state.get("retry_count", 0) <= MAX_RETRIES
    ):
        return "analyst"
    return END


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------

class MultiAgent(BaseAgent):
    """Three-agent pipeline: Planner → Analyst → Reviewer with retry loop."""

    def __init__(self, model: str = "gpt-4o", provider: str | None = None, temperature: float = 0.2):
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self._graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        plan_node = _build_planner(self.model, self.temperature, self.provider)
        analyst_node = _build_analyst(self.model, self.temperature, self.provider)
        review_node = _build_reviewer(self.model, self.temperature, self.provider)

        graph = StateGraph(MultiAgentState)
        graph.add_node("planner", plan_node)
        graph.add_node("analyst", analyst_node)
        graph.add_node("reviewer", review_node)

        graph.add_edge(START, "planner")
        graph.add_edge("planner", "analyst")
        graph.add_edge("analyst", "reviewer")
        graph.add_conditional_edges("reviewer", _should_retry, ["analyst", END])

        return graph.compile()

    def analyze(self, question: str, dataset_name: str) -> AnalysisResult:
        info = get_dataset_info(dataset_name)

        # Get schema for the planner (lightweight, no full load)
        from src.data.loader import DatasetLoader
        loader = DatasetLoader()
        schema = loader.get_schema(dataset_name, sample_rows=3)
        all_cols = schema["columns"]
        var_labels = get_variable_labels(dataset_name)

        limit = 100
        shown_cols = all_cols[:limit]
        lines = []
        for c in shown_cols:
            label = var_labels.get(c) or ""
            label_str = f" — {label}" if label else ""
            lines.append(f"  {c} ({schema['dtypes'][c]}){label_str}")
        schema_str = "\n".join(lines)
        if len(all_cols) > limit:
            schema_str += f"\n  ... and {len(all_cols) - limit} more columns. Use search_columns or get_variable_info tools to find and inspect specific ones."

        initial_state: MultiAgentState = {
            "question": question,
            "dataset_name": info.name,
            "dataset_description": info.description,
            "weight_column": info.weight_column,
            "schema_info": schema_str,
            "plan": "",
            "analyst_messages": [],
            "code_executed": "",
            "analysis_output": "",
            "review": "",
            "retry_count": 0,
            "final_answer": "",
        }

        t0 = time.time()
        try:
            result_state = self._graph.invoke(initial_state, {"recursion_limit": 60})
            elapsed = time.time() - t0

            return AnalysisResult(
                question=question,
                dataset=dataset_name,
                answer=result_state.get("final_answer", result_state.get("review", "")),
                code_executed=result_state.get("code_executed", ""),
                raw_statistics={"plan": result_state.get("plan", "")},
                execution_time_seconds=round(elapsed, 2),
                retries=result_state.get("retry_count", 0),
            )
        except Exception as e:
            elapsed = time.time() - t0
            return AnalysisResult(
                question=question,
                dataset=dataset_name,
                answer="",
                execution_time_seconds=round(elapsed, 2),
                errors=[str(e)],
            )
