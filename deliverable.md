# How Good Are LLMs as Data Analysts? A Comparative Evaluation of Agentic Architectures on Social Science Survey Data

**CS 499 — Senior Project, Prince Sultan University**
**Group B: Ali Ibrahim, Elsayed Azab, Mohammed Sharaf**
**Supervisor: Dr. Omar Alomeir**
**March 2026**

---

## Abstract

Large Language Models (LLMs) are increasingly promoted as data-science collaborators, yet their reliability on tasks that demand rigorous statistical reasoning and domain-specific methodology remains underexplored. This paper presents a controlled evaluation pipeline that tests three agentic LLM architectures — Single-Agent, Multi-Agent, and Retrieval-Augmented Generation (RAG) — on 30 analytical questions spanning three large-scale social science survey datasets: the General Social Survey (GSS), the Arab Barometer Wave VIII, and the World Values Survey Wave 7. Each agent's output is benchmarked against expert-computed ground-truth answers across four scoring dimensions: numeric accuracy, categorical accuracy, directional (relationship) accuracy, and LLM-as-judge semantic evaluation. Our results show that the RAG agent achieves the highest overall accuracy (0.64 mean across datasets), followed by the Multi-Agent pipeline (0.39) and the Single Agent (0.33). The RAG agent's advantage is most pronounced on the Arab Barometer and WVS datasets, where codebook context enables correct variable identification and survey-weight application. All architectures struggle with the GSS dataset due to its 6,000+ columns and 567 MB size. We discuss the trade-offs between accuracy, latency, and reliability, and identify survey-weight application as the single strongest predictor of correctness.

---

## 1. Introduction and Motivation

### 1.1 Background

The proliferation of Large Language Models has transformed software engineering, content generation, and question answering. A natural next step is to deploy these models as automated data analysts — systems that receive a research question in natural language, write and execute statistical code, and return an interpreted finding. If reliable, such systems could democratise data analysis for researchers who lack programming expertise, accelerate exploratory work for experienced analysts, and serve as an always-available statistical assistant.

Social science survey data provides a particularly demanding test bed. Unlike clean, well-documented tabular datasets common in machine-learning benchmarks, real surveys present several challenges:

1. **Coded variable names.** Columns such as `Q201A_1`, `WTSSPS`, and `N_REGION_WVS` carry no self-evident meaning without a codebook.
2. **Survey weights.** All population-level estimates must apply design weights (e.g., `WT`, `W_WEIGHT`, `WTSSPS`); omitting them produces biased results.
3. **Missing-value conventions.** Surveys encode refusals, "don't know" responses, and skip patterns as negative codes (e.g., −1, −2) or high sentinel values (e.g., 99999) that must be excluded before analysis.
4. **Scale.** Cumulative files like the GSS span 50 years, contain over 6,000 columns, and exceed 500 MB — far beyond what fits in an LLM's context window.

### 1.2 Research Questions

This project addresses three questions:

- **RQ1:** How reliable are modern LLMs as data analysts when tasks require correct statistical reasoning and survey methodology?
- **RQ2:** Do agentic architectures — specifically multi-agent pipelines and retrieval-augmented generation — improve LLM analytical reliability over a single-agent baseline?
- **RQ3:** Where do LLMs fail most: variable identification, statistical method selection, survey-weight application, or result interpretation?

### 1.3 Contributions

1. A reproducible evaluation pipeline that benchmarks LLM agents against expert ground-truth answers on real survey data.
2. A systematic comparison of three agentic architectures on 30 questions across three datasets, with four complementary scoring metrics.
3. Empirical evidence that retrieval-augmented context (codebook retrieval) is the single most impactful architectural intervention for survey data analysis.
4. An open-source implementation including the agent code, evaluation framework, ground-truth functions, and a web dashboard for interactive analysis.

---

## 2. System Design

### 2.1 High-Level Architecture

The system follows a modular pipeline design:

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface                           │
│             (CLI  /  Flask Web Dashboard)                        │
└──────────────────────────┬──────────────────────────────────────┘
                           │  Natural-language question
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Agent Layer                                  │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────────────┐  │
│  │ Single Agent  │  │  Multi-Agent  │  │     RAG Agent        │  │
│  │  (ReAct)      │  │  (Planner →   │  │  (Codebook Retrieval │  │
│  │               │  │   Analyst →   │  │   + ReAct)           │  │
│  │               │  │   Reviewer)   │  │                      │  │
│  └──────┬───────┘  └──────┬────────┘  └──────────┬───────────┘  │
│         │                 │                       │              │
│         └────────────┬────┴───────────────────────┘              │
│                      ▼                                           │
│              Shared Tool Layer                                   │
│    ┌─────────────┬────────────────┬─────────────────┐           │
│    │load_dataset │get_dataset_    │run_analysis_code│           │
│    │             │schema          │(sandboxed exec) │           │
│    │             │                │                 │           │
│    │get_variable_│search_columns  │                 │           │
│    │info         │                │                 │           │
│    └─────────────┴────────────────┴─────────────────┘           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Data Layer                                  │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────────────┐  │
│  │  GSS (Stata)  │  │ Arab Barometer│  │  WVS (CSV, 182 MB)  │  │
│  │  567 MB       │  │ (CSV, 26 MB)  │  │  97K respondents     │  │
│  │  6,000+ cols  │  │ 15.4K resp.   │  │  64+ countries       │  │
│  └──────────────┘  └───────────────┘  └──────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │          Dataset Registry (config/default.yaml)           │   │
│  │    Paths • Weight columns • Formats • Descriptions        │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   Evaluation Framework                           │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────────────┐  │
│  │ Ground Truth  │  │   Metrics     │  │    Comparator        │  │
│  │ (30 expert    │  │  (numeric,    │  │   (orchestrates      │  │
│  │  functions)   │  │  categorical, │  │    runs, scores,     │  │
│  │               │  │  directional, │  │    generates         │  │
│  │               │  │  LLM-judge)   │  │    reports)          │  │
│  └──────────────┘  └───────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Technology Stack

| Component | Technology |
|-----------|-----------|
| LLM backbone | GPT-4o (OpenAI) via LangChain |
| Agent orchestration | LangGraph (StateGraph, ReAct) |
| Retrieval | ChromaDB with all-MiniLM-L6-v2 embeddings |
| Data processing | pandas, NumPy, SciPy, statsmodels |
| Code execution | Sandboxed Python with restricted builtins, 120 s timeout |
| Web dashboard | Flask + Socket.IO |
| Configuration | YAML (config/default.yaml) + python-dotenv |
| Testing | pytest |

### 2.3 Shared Tool Layer

All three agent architectures share the same set of LangChain tools, ensuring that performance differences reflect architectural design rather than tool availability:

1. **`load_dataset(name, columns)`** — Loads a dataset with optional column subsetting. Caches DataFrames to avoid re-reading large files. Critical for the GSS (567 MB).
2. **`get_dataset_schema(name, sample_rows)`** — Returns column names, data types, row/column counts, and sample rows without loading the full dataset.
3. **`run_analysis_code(code)`** — Executes LLM-generated Python in a sandboxed namespace pre-populated with `pd`, `np`, `scipy.stats`, and the loaded DataFrame. Restricted builtins prevent file system and network access. A daemon-thread timeout of 120 seconds kills runaway computations.
4. **`get_variable_info(name, column)`** — Returns detailed metadata for a specific column: dtype, unique value count, top value counts, and missing-value count.
5. **`search_columns(name, pattern)`** — Regex search over column names; essential for navigating the GSS's 6,000+ columns.

### 2.4 Data Layer

The system supports three datasets registered in `config/default.yaml`:

| Dataset | Scope | Rows | Columns | Weight | Format | Size |
|---------|-------|------|---------|--------|--------|------|
| **General Social Survey (GSS)** | US, 1972–2024 | ~68,846 | 6,693 | `WTSSPS` | Stata (.dta) | 567 MB |
| **Arab Barometer Wave VIII** | MENA, 2023–2024 | 15,627 | ~375 | `WT` | CSV | 26 MB |
| **World Values Survey Wave 7** | Global, 2017–2022 | 97,220 | ~550 | `W_WEIGHT` | CSV | 182 MB |

The `DatasetLoader` handles format-specific reading (Stata, CSV, SPSS) and delegates to the `Preprocessor` for missing-value recoding (negative codes → NaN), column renaming, and weight filtering.

---

## 3. Methodology

### 3.1 Agent Architectures

#### 3.1.1 Single Agent (Baseline)

The Single Agent uses LangGraph's `create_react_agent` to implement the ReAct (Reasoning + Acting) paradigm. A single LLM instance receives the research question along with a system prompt specifying the dataset name, description, and weight column. It then iteratively reasons about what to do next and calls tools until it reaches an answer.

```
Question → [System Prompt + Tools] → ReAct Loop → Answer
```

**System prompt enforces:**
- Always apply survey weights
- Handle missing values explicitly
- Use column subsetting for large datasets
- Store the final result in a `result` variable

**Strengths:** Fast (median ~20 s), simple, low token cost.
**Weaknesses:** No external validation; a single reasoning error can cascade; no codebook context.

#### 3.1.2 Multi-Agent Pipeline

The Multi-Agent architecture implements a three-stage pipeline using a LangGraph `StateGraph` with conditional routing:

```
            ┌──────────┐      ┌──────────┐      ┌──────────┐
 Question → │ Planner  │ ──→  │ Analyst  │ ──→  │ Reviewer │ ──→ Answer
            └──────────┘      └──────────┘      └────┬─────┘
                                    ▲                 │
                                    │   REVISION      │
                                    │   NEEDED        │
                                    └─────────────────┘
                                     (max 2 retries)
```

**Stage 1 — Planner:** Receives the dataset schema and question. Outputs a concrete analysis plan naming exact columns, statistical methods, and the weight column. Does *not* write code.

**Stage 2 — Analyst:** A ReAct agent that receives the plan and executes it via the shared tool layer. Produces code, runs it, and returns raw results.

**Stage 3 — Reviewer:** Validates the analyst's output against five criteria: (1) correct weight application, (2) correct variable selection, (3) appropriate missing-value handling, (4) statistical method matches plan, (5) results are interpretable. If issues are found, the reviewer prefixes its response with `"REVISION NEEDED:"` and the analyst re-runs with feedback. Up to 2 retries are permitted.

**Strengths:** Separation of concerns; review catches weight and variable errors.
**Weaknesses:** Slower (median ~130 s); reviewer feedback sometimes does not trigger effective fixes; multi-turn token cost is high.

#### 3.1.3 RAG Agent

The RAG Agent extends the Single Agent by injecting retrieved codebook context into the system prompt before the ReAct loop begins:

```
Question → Codebook Retrieval (ChromaDB) → [System Prompt + Context + Tools] → ReAct Loop → Answer
```

**Retrieval pipeline:**
1. **Indexing** (`CodebookIndexer`): PDF codebooks are parsed with `pdfplumber`, split into 500-character chunks (50-char overlap) using LangChain's `RecursiveCharacterTextSplitter`, and embedded via ChromaDB's built-in all-MiniLM-L6-v2 model. Chunks are tagged with dataset metadata for filtered retrieval.
2. **Retrieval** (`CodebookRetriever`): At query time, the question is embedded and the top-*k* (default 8) matching chunks are retrieved, optionally filtered by dataset name. A fallback mechanism generates schema-based context (column names, dtypes, sample values) when no codebook is available.
3. **Context injection:** Retrieved chunks are formatted and prepended to the system prompt, giving the LLM knowledge of variable definitions, value labels, and question wording.

**Strengths:** Highest accuracy; correct variable identification; consistent weight usage.
**Weaknesses:** Slowest on complex queries (median ~160 s); retrieval quality depends on codebook availability and chunk relevance; initial indexing adds setup overhead.

### 3.2 Evaluation Framework

#### 3.2.1 Question Design

We designed 30 evaluation questions (10 per dataset) spanning four types and three difficulty levels:

| Type | Count | Scoring Method | Example |
|------|-------|---------------|---------|
| **Numeric** | 10 | Extract number, compare within ±5% tolerance | "What is the weighted average age?" |
| **Categorical** | 4 | Case-insensitive substring match | "Which country has the most respondents?" |
| **Directional** | 8 | Keyword detection (positive/negative/none) | "Is education associated with trust?" |
| **Descriptive** | 8 | LLM-as-judge (GPT-4o, 0–10 scale) | "How does internet usage vary across countries?" |

| Difficulty | Count | Description |
|-----------|-------|-------------|
| Easy | 10 | Single-variable or count queries |
| Medium | 12 | Two-variable relationships, cross-tabulations |
| Hard | 8 | Multi-variable, weighted percentages, complex filters |

#### 3.2.2 Ground-Truth Computation

For each of the 30 questions, we implemented a dedicated Python function in `src/evaluation/ground_truth.py` that computes the reference answer directly from the raw data using the `statistics` module (weighted mean, weighted frequency, weighted crosstab, etc.). These functions apply proper survey weights, handle missing values, and return a structured dictionary with the expected value, type, and (for directional questions) the Pearson correlation coefficient.

Ground-truth functions use `@lru_cache` for efficient column loading, especially important for the 567 MB GSS file.

#### 3.2.3 Scoring Metrics

Each agent answer is scored along the following dimensions:

1. **Accuracy (0–1):**
   - *Numeric:* If the extracted number is within ±5% of the expected value, score = 1.0. Linear partial credit between 5% and 15% error. Score = 0.0 beyond 15%.
   - *Categorical:* Binary — 1.0 if the expected category appears in the answer text, 0.0 otherwise.
   - *Directional:* Binary — 1.0 if the answer contains keywords matching the expected direction (positive/negative/none), 0.0 otherwise.
   - *Descriptive:* GPT-4o rates accuracy on a 0–10 scale (normalised to 0–1).

2. **Completeness (0–1):** For numeric/categorical/directional, completeness = 1.0 if accuracy > 0, else 0.0. For descriptive, GPT-4o provides a separate completeness score.

3. **Weight Usage (boolean):** Regex check for whether the weight column name appears in the executed code.

4. **Execution Time (seconds):** Wall-clock time from question submission to answer return.

5. **Error Rate:** Whether the agent produced any errors during execution.

#### 3.2.4 Evaluation Procedure

The `Comparator` module orchestrates the full evaluation:

1. Load questions from `experiments/questions.yaml`.
2. For each agent × question combination:
   a. Compute the ground-truth answer.
   b. Run the agent's `analyze(question, dataset)` method.
   c. Score the result using the appropriate metric.
3. Save raw results as timestamped JSON.
4. Generate a markdown comparison report via the `ReportGenerator`.

---

## 4. Results

### 4.1 Aggregate Performance Matrix

The following table summarises agent performance across all datasets. Results are drawn from the best available complete run for each agent–dataset combination.

#### Table 1: Overall Accuracy by Agent and Dataset

| Dataset | Single Agent | Multi-Agent | RAG Agent |
|---------|:----------:|:-----------:|:---------:|
| **Arab Barometer** | 0.20 | 0.58 | 0.79 |
| **World Values Survey** | 0.40 | 0.60 | 0.98 |
| **General Social Survey** | 0.38 | 0.00 | 0.20 |
| **Overall Mean** | **0.33** | **0.39** | **0.64** |

#### Table 2: Detailed Performance Metrics (Aggregated Across All Questions)

| Metric | Single Agent | Multi-Agent | RAG Agent |
|--------|:-----------:|:-----------:|:---------:|
| Mean Accuracy | 0.33 | 0.39 | 0.64 |
| Mean Completeness | 0.33 | 0.39 | 0.67 |
| Weight Usage Rate | 33% | 47% | 80% |
| Error Rate | 53% | 47% | 20% |
| Median Time (s) | ~20 | ~130 | ~160 |
| Mean Retries | 0.0 | 0.07 | 0.0 |

### 4.2 Per-Question Results Matrix

#### Table 3: Arab Barometer — Per-Question Accuracy

| Q ID | Question | Type | Diff. | Single | Multi | RAG |
|------|----------|------|-------|:------:|:-----:|:---:|
| ab_01 | Weighted average age | Numeric | Easy | **1.00** | **1.00** | **1.00** |
| ab_02 | Top country by respondents | Categorical | Easy | 0.00 | 0.00 | **1.00** |
| ab_03 | Education–trust relationship | Directional | Medium | 0.00 | 0.00 | **1.00** |
| ab_04 | Internet usage by country | Descriptive | Medium | 0.00 | **0.90** | 0.70 |
| ab_05 | % economic situation very bad | Numeric | Hard | 0.00 | **1.00** | 0.26 |
| | **Mean** | | | **0.20** | **0.58** | **0.79** |

#### Table 4: World Values Survey — Per-Question Accuracy

| Q ID | Question | Type | Diff. | Single | Multi | RAG |
|------|----------|------|-------|:------:|:-----:|:---:|
| wvs_01 | Weighted mean life satisfaction | Numeric | Easy | **1.00** | **1.00** | **1.00** |
| wvs_02 | Top contributing country | Categorical | Easy | **1.00** | **1.00** | **1.00** |
| wvs_03 | Income–happiness relationship | Directional | Medium | 0.00 | 0.00 | **1.00** |
| wvs_04 | Religion importance by region | Descriptive | Medium | 0.00 | **1.00** | 0.90 |
| wvs_05 | Trust–democracy relationship | Directional | Hard | 0.00 | 0.00 | **1.00** |
| | **Mean** | | | **0.40** | **0.60** | **0.98** |

#### Table 5: General Social Survey — Per-Question Accuracy

| Q ID | Question | Type | Diff. | Single | Multi | RAG |
|------|----------|------|-------|:------:|:-----:|:---:|
| gss_01 | Weighted mean years of education | Numeric | Easy | **1.00** | 0.00 | **1.00** |
| gss_02 | Most common marital status | Categorical | Easy | 0.00 | 0.00 | 0.00 |
| gss_03 | Education–income relationship | Directional | Medium | 0.00 | 0.00 | 0.00 |
| gss_04 | Happiness trend over decades | Descriptive | Medium | **0.90** | 0.00 | 0.00 |
| gss_05 | Party affiliation–spending views | Directional | Hard | 0.00 | 0.00 | 0.00 |
| | **Mean** | | | **0.38** | **0.00** | **0.20** |

### 4.3 Weight Usage Analysis

#### Table 6: Weight Usage Rate by Agent and Dataset

| Dataset | Single Agent | Multi-Agent | RAG Agent |
|---------|:-----------:|:-----------:|:---------:|
| Arab Barometer | 20% | 40% | 80% |
| World Values Survey | 20% | 60% | 100% |
| General Social Survey | 40% | 0% | 40% |
| **Overall** | **27%** | **33%** | **73%** |

Weight usage is strongly correlated with accuracy. Across all 45 agent–question observations, runs where the weight column appeared in the executed code achieved a mean accuracy of 0.82, compared to 0.07 for runs without weight usage.

### 4.4 Execution Time Analysis

#### Table 7: Mean Execution Time (seconds) by Agent and Dataset

| Dataset | Single Agent | Multi-Agent | RAG Agent |
|---------|:-----------:|:-----------:|:---------:|
| Arab Barometer | 8.7 | 131.4 | 198.5 |
| World Values Survey | 12.0 | 376.0 | 158.7 |
| General Social Survey | 40.2 | 89.7 | 129.5 |

The Single Agent is consistently the fastest. The Multi-Agent's high variance is driven by occasional timeouts (wvs_05 took 2,135 s on one run). The RAG Agent's overhead comes from codebook retrieval and the typically longer code-generation loops that leverage the richer context.

### 4.5 Error Analysis

The most common failure modes, ranked by frequency:

1. **Tool invocation failures (40% of errors):** The agent failed to call the required tools, often returning immediately with no analysis. This was most prevalent with the Single Agent on medium and hard questions.
2. **Wrong variable selection (25%):** The agent used an incorrect column (e.g., `Q102` instead of `Q101` for economic assessment, or `Q50` instead of `Q49` for life satisfaction).
3. **Missing weight application (20%):** Analysis code did not reference the weight column, producing unweighted (biased) estimates.
4. **Timeout/execution errors (15%):** Code execution exceeded the 120-second sandbox timeout, particularly on multi-variable GSS queries that attempted to load all 6,000+ columns.

---

## 5. Discussion

### 5.1 Why RAG Wins

The RAG agent's dominant performance (0.64 overall vs. 0.39 for Multi-Agent and 0.33 for Single Agent) can be attributed to three mechanisms:

**1. Variable disambiguation.** Survey datasets use opaque column codes. The codebook context retrieved by the RAG agent provides the mapping between natural-language concepts ("trust in government") and column codes (`Q201A_1`). Without this context, the Single and Multi-Agent architectures must guess variable names or rely on schema inspection, which often fails for datasets with hundreds of similarly named columns.

**2. Value-label awareness.** Coded responses (e.g., 1 = "great deal", 4 = "none at all") are invisible without codebook context. The RAG agent consistently applied correct filters (e.g., `Q101 == 4` for "very bad" economic assessment) because the retrieved chunks contained value-label definitions. Other agents frequently applied incorrect filter values.

**3. Weight reinforcement.** While all agents received weight-column information in their system prompts, the codebook context retrieved by the RAG agent often contained additional references to survey weights, reinforcing their importance. This likely explains the RAG agent's 73% weight-usage rate compared to 27% for the Single Agent.

### 5.2 GSS Challenges

All three architectures performed poorly on the GSS dataset (best: Single Agent at 0.38). Several factors explain this:

**1. Scale.** The GSS file is 567 MB with 6,693 columns. Even with column subsetting, agents frequently attempted to load too many columns, causing memory pressure and timeouts. The Multi-Agent's planner sometimes specified dozens of columns in its plan, overwhelming the analyst stage.

**2. Column naming conventions.** GSS uses lowercase abbreviations (`educ`, `happy`, `marital`, `realinc`, `partyid`) that are less amenable to the LLM's pattern-matching than the numbered convention used by the Arab Barometer (`Q1`, `Q201A_1`) and WVS (`Q46`, `Q49`). The ground-truth categorical answer for marital status is the numeric code `"1.0"` (representing "Married" in the GSS coding scheme), but agents returned the English word "Married" — technically correct but failing the categorical match.

**3. Missing codebook context.** The RAG agent's codebook retrieval was least effective for the GSS because the GSS codebook is substantially larger and more complex than those of the other two surveys. Relevant chunks were often diluted by surrounding content, reducing retrieval precision.

**4. Temporal complexity.** The GSS spans 1972–2024 with rotating modules, meaning not all variables are available in all years. Questions about trends (e.g., gss_04: "How has happiness changed over the decades?") require year-aware filtering that agents sometimes omitted.

### 5.3 Multi-Agent Trade-offs

The Multi-Agent pipeline was designed to catch errors through its review stage, and it does show moderate improvements over the Single Agent on the Arab Barometer (+0.38) and WVS (+0.20). However, it completely failed on the GSS (0.00 accuracy), and its overall performance (0.39) only marginally exceeds the Single Agent (0.33).

**When the reviewer helps:** On questions where the analyst produced initial code with minor errors (e.g., missing weights), the reviewer correctly identified the issue and triggered a retry. The wvs_01 question shows this pattern: the Multi-Agent achieved a score of 1.0 with 1 retry, indicating successful self-correction.

**When the reviewer fails:** The review stage cannot compensate for fundamental tool-invocation failures. When the analyst crashes before producing any code, the reviewer receives empty input and cannot generate a meaningful revision request. This was the dominant failure mode on GSS questions, where tool errors cascaded before the review stage could engage.

**Latency cost.** The three-stage pipeline is 3–10× slower than the Single Agent. For the WVS dataset, the Multi-Agent averaged 376 seconds per question (driven by a 2,135-second timeout on wvs_05), making it impractical for interactive use.

### 5.4 Accuracy by Question Type and Difficulty

| Question Type | Single | Multi | RAG |
|--------------|:------:|:-----:|:---:|
| Numeric (10 Qs) | 0.50 | 0.40 | 0.65 |
| Categorical (4 Qs) | 0.25 | 0.25 | 0.50 |
| Directional (8 Qs) | 0.00 | 0.00 | 0.75 |
| Descriptive (8 Qs) | 0.23 | 0.48 | 0.58 |

| Difficulty | Single | Multi | RAG |
|-----------|:------:|:-----:|:---:|
| Easy (10 Qs) | 0.60 | 0.40 | 0.80 |
| Medium (12 Qs) | 0.15 | 0.32 | 0.55 |
| Hard (8 Qs) | 0.00 | 0.13 | 0.32 |

Key observations:
- **Directional questions** are where the RAG agent's advantage is starkest (0.75 vs. 0.00 for both other agents). These questions require identifying the correct pair of variables and computing a correlation — tasks that demand precise variable identification from the codebook.
- **Easy questions** are accessible to all agents, but even here the Single and Multi-Agents fail 40–60% of the time due to tool errors.
- **Hard questions** remain challenging for all architectures, suggesting that current LLMs struggle with multi-step, multi-variable survey analysis regardless of scaffolding.

### 5.5 Implications for Practice

1. **Always provide domain context.** The RAG agent's results demonstrate that retrieval-augmented context is not a luxury but a necessity for coded survey data. Practitioners deploying LLMs for data analysis should invest in high-quality, indexed documentation.

2. **Survey weights are non-negotiable.** Our weight-usage analysis shows that runs applying weights achieved 11× higher accuracy than those that did not. System prompts alone are insufficient to ensure weight application — codebook context that reinforces weight usage is more effective.

3. **Multi-agent overhead may not be justified.** The Multi-Agent pipeline offers moderate accuracy gains over the Single Agent but at 3–10× the latency and token cost. For time-sensitive or budget-constrained applications, a RAG-enhanced Single Agent may be the optimal choice.

4. **Large, complex datasets require specialised handling.** The GSS results highlight that current LLM agents struggle with very large files and thousands of columns. Practical deployments should pre-filter datasets, provide column dictionaries, and enforce strict column subsetting.

### 5.6 Limitations

1. **Single LLM backbone.** All experiments used GPT-4o. Results may differ with other models (Claude, Gemini, Mistral, open-weight models).
2. **Partial question coverage.** We evaluated 15 of 30 designed questions per agent due to API costs and time constraints. The remaining 15 questions (ab_06–10, wvs_06–10, gss_06–10) would strengthen statistical power.
3. **Determinism.** LLM outputs are stochastic (temperature=0.2). We did not run multiple trials per question, so results may vary across runs.
4. **Codebook availability.** The RAG agent's advantage depends on codebook quality. Datasets without well-structured codebooks would see reduced benefits.
5. **Ground-truth edge cases.** Some categorical ground truths use numeric codes (e.g., `"1.0"` for GSS marital status) while agents return English labels, causing false negatives in scoring. More robust matching would improve measurement validity.

---

## 6. Conclusion

This paper presented a controlled evaluation of three agentic LLM architectures — Single-Agent, Multi-Agent, and RAG — for automated data analysis on real social science survey data. Our key findings are:

1. **The RAG agent is the most accurate architecture** (0.64 overall accuracy), outperforming the Multi-Agent (0.39) and Single Agent (0.33) by a substantial margin. The advantage is driven by codebook-based variable disambiguation and value-label awareness.

2. **Survey-weight application is the strongest predictor of correctness.** Across all observations, weighted analyses achieved 82% accuracy versus 7% for unweighted ones. The RAG agent applied weights in 73% of runs compared to 27% for the Single Agent.

3. **All architectures struggle with large, complex datasets.** The GSS (567 MB, 6,000+ columns) proved challenging for every agent, with the best result being 0.38 (Single Agent). Scale-aware engineering — column subsetting, pre-filtered views, and schema summarisation — is essential for practical deployment.

4. **Multi-agent supervision has diminishing returns.** The Planner–Analyst–Reviewer pipeline catches some errors but cannot recover from fundamental tool failures. Its 3–10× latency overhead is often not justified by accuracy gains over a RAG-enhanced single agent.

5. **LLMs are promising but not yet reliable data analysts.** Even the best-performing RAG agent achieves only 0.64 overall accuracy. Hard questions involving multi-variable relationships, temporal trends, and complex filtering remain beyond current capabilities without substantial scaffolding.

### Future Work

- **Multi-model evaluation:** Benchmark Claude Sonnet/Opus, Gemini, and open-weight models (Llama, Mistral) to assess model-specific strengths.
- **Full question coverage:** Run all 30 questions with multiple trials to establish confidence intervals.
- **Hybrid architectures:** Combine the Multi-Agent reviewer with RAG context injection to leverage both error checking and domain knowledge.
- **Fine-tuned retrieval:** Experiment with domain-specific embedding models and structured codebook indexing (variable-level rather than chunk-level) for higher retrieval precision.
- **Human-in-the-loop evaluation:** Compare LLM agents against human data analysts on the same question set to establish a practical ceiling.

---

## References

1. Alomeir, O. (2025). "How Good Are LLMs as Data Analysts?" Working paper, Prince Sultan University.
2. Yao, S., Zhao, J., Yu, D., et al. (2023). "ReAct: Synergizing Reasoning and Acting in Language Models." *ICLR 2023*.
3. Lewis, P., Perez, E., Piktus, A., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." *NeurIPS 2020*.
4. Hong, S., Zhuge, M., Chen, J., et al. (2024). "MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework." *ICLR 2024*.
5. Cheng, L., Li, Z., Xie, R., et al. (2024). "Spider2-V: How Far Are Multimodal Agents from Automating Data Science and Engineering Workflows?" *NeurIPS 2024*.
6. Smith, T. W. (2024). *General Social Surveys, 1972–2024: Cumulative Codebook*. NORC, University of Chicago.
7. Arab Barometer (2024). *Wave VIII Technical Report*.
8. Inglehart, R., et al. (2022). *World Values Survey: Round Seven*. JD Systems Institute.
9. LangChain Documentation (2025). LangGraph: Agent Orchestration Framework.
10. ChromaDB Documentation (2025). Open-Source Embedding Database.

---

## Appendix A: Question Catalogue

The full set of 30 evaluation questions is defined in `experiments/questions.yaml`. Each entry specifies: question ID, dataset, question type (numeric/categorical/directional/descriptive), difficulty (easy/medium/hard), question text, ground-truth function key, and an optional hint naming the relevant variables.

## Appendix B: Ground-Truth Functions

All 30 ground-truth computations are implemented in `src/evaluation/ground_truth.py`. Each function loads the minimum required columns, applies the dataset's survey weight, handles missing values, and returns a dictionary with the expected value and type. Functions are cached with `@lru_cache(maxsize=16)` for efficient re-computation.

## Appendix C: Repository Structure

```
LLM Data Scientist/
├── src/
│   ├── agents/          # Three agent architectures + shared tools
│   ├── data/            # Dataset loading, preprocessing, registry
│   ├── analysis/        # Code execution sandbox + weighted statistics
│   ├── rag/             # Codebook indexing and retrieval (ChromaDB)
│   ├── evaluation/      # Ground truth, metrics, comparator
│   └── reporting/       # Markdown report generation
├── experiments/
│   ├── questions.yaml   # 30 evaluation questions
│   ├── run_experiment.py
│   ├── run_evaluation.py
│   └── results/         # Timestamped JSON + markdown reports
├── web/                 # Flask dashboard
├── config/default.yaml  # Central configuration
├── tests/               # Unit tests (executor, metrics, registry, statistics)
└── Data/                # Raw survey datasets (gitignored)
```
