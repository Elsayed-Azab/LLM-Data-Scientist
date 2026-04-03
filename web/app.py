"""Flask backend for the LLM Data Scientist dashboard."""

import hashlib
import json
import sys
import time
import queue
from dataclasses import asdict
from pathlib import Path
from threading import Thread

from flask import Flask, render_template, request, jsonify, Response
from flask_socketio import SocketIO
from dotenv import load_dotenv

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

from src.agents.single_agent import SingleAgent
from src.agents.multi_agent import MultiAgent
from src.agents.rag_agent import RAGAgent
from src.agents.base import AnalysisResult
from src.data.registry import list_datasets, get_dataset_info
from src.evaluation.metrics import check_weight_usage
from src.evaluation.comparator import load_questions, evaluate_answer
from src.evaluation.ground_truth import compute_ground_truth


# ── Web result cache ────────────────────────────────────────────────

_WEB_CACHE_DIR = Path("experiments/.cache/web_results")


def _web_cache_key(agent_type: str, model: str, dataset: str, question: str) -> str:
    """Build a deterministic cache key from the request parameters."""
    q_hash = hashlib.md5(question.encode()).hexdigest()[:12]
    safe_model = model.replace("/", "_").replace(":", "_")
    return f"{agent_type}_{safe_model}_{dataset}_{q_hash}"


def _save_web_cache(key: str, result: AnalysisResult) -> None:
    _WEB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _WEB_CACHE_DIR / f"{key}.json"
    with open(path, "w") as f:
        json.dump(asdict(result), f, indent=2, default=str)


def _load_web_cache(key: str) -> AnalysisResult | None:
    path = _WEB_CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        return AnalysisResult(
            question=data.get("question", ""),
            dataset=data.get("dataset", ""),
            answer=data.get("answer", ""),
            code_executed=data.get("code_executed", ""),
            raw_statistics=data.get("raw_statistics", {}),
            execution_time_seconds=data.get("execution_time_seconds", 0.0),
            errors=data.get("errors", []),
            retries=data.get("retries", 0),
        )
    except Exception:
        return None

app = Flask(__name__)
app.config["SECRET_KEY"] = "llm-data-scientist"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

AGENT_CLASSES = {
    "single": SingleAgent,
    "multi": MultiAgent,
    "rag": RAGAgent,
}

SUGGESTED_QUESTIONS = {
    "gss": [
        "What is the weighted average years of education for GSS respondents?",
        "What is the most common marital status among respondents?",
        "Is there a relationship between education and income?",
        "How has general happiness changed over the decades?",
    ],
    "arab_barometer": [
        "What is the weighted average age of respondents?",
        "Which country has the most respondents?",
        "Is there a relationship between education level and trust in government?",
        "How does internet usage vary across countries?",
    ],
    "wvs": [
        "What is the weighted mean life satisfaction score?",
        "Which country contributed the most respondents?",
        "Is there a relationship between income level and happiness?",
        "How does importance of religion vary by world region?",
    ],
}

# ── Page routes ──────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/analysis")
def analysis():
    return render_template("analysis.html")

@app.route("/comparison")
def comparison():
    return render_template("comparison.html")

@app.route("/results")
def results():
    return render_template("results.html")

@app.route("/matrix")
def matrix():
    return render_template("matrix.html")

@app.route("/models")
def models():
    return render_template("models.html")

# ── API routes ───────────────────────────────────────────────────────

@app.route("/api/datasets")
def api_datasets():
    datasets = []
    for name in list_datasets():
        info = get_dataset_info(name)
        datasets.append({
            "name": name,
            "description": info.description,
            "weight_column": info.weight_column,
            "format": info.format,
        })
    return jsonify(datasets)

@app.route("/api/questions/<dataset>")
def api_questions(dataset):
    return jsonify(SUGGESTED_QUESTIONS.get(dataset, []))

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.json
    question = data.get("question", "").strip()
    dataset = data.get("dataset", "")
    agent_type = data.get("agent", "single")
    model = data.get("model", "gpt-4o")
    provider = data.get("provider")
    temperature = data.get("temperature", 0.2)

    if not question or not dataset:
        return jsonify({"error": "Please enter a question and select a dataset."}), 400
    if agent_type not in AGENT_CLASSES:
        return jsonify({"error": f"Unknown agent type: {agent_type}. Choose single, multi, or rag."}), 400

    # Check cache
    cache_key = _web_cache_key(agent_type, model, dataset, question)
    cached = _load_web_cache(cache_key)
    if cached is not None:
        ds_info = get_dataset_info(dataset)
        weight_used = check_weight_usage(cached.code_executed, ds_info.weight_column)
        return jsonify({
            "answer": cached.answer,
            "time": cached.execution_time_seconds,
            "retries": cached.retries,
            "weights_used": weight_used,
            "errors": cached.errors,
            "success": cached.success,
            "code_executed": cached.code_executed,
            "raw_statistics": cached.raw_statistics,
            "cached": True,
        })

    try:
        agent = AGENT_CLASSES[agent_type](model=model, provider=provider, temperature=temperature)
        result = agent.analyze(question, dataset)
    except Exception as e:
        err = str(e)
        if "api_key" in err.lower() or "API key" in err:
            err = f"Missing API key. Set your {'ANTHROPIC_API_KEY' if provider == 'anthropic' else 'OPENAI_API_KEY'} in .env"
        elif "rate_limit" in err.lower():
            err = f"Rate limit exceeded for {model}. Wait a moment or switch to a different model."
        elif "timeout" in err.lower() or "timed out" in err.lower():
            err = f"Request timed out. The {model} model took too long to respond."
        return jsonify({"error": err}), 500

    # Save to cache
    _save_web_cache(cache_key, result)

    ds_info = get_dataset_info(dataset)
    weight_used = check_weight_usage(result.code_executed, ds_info.weight_column)

    return jsonify({
        "answer": result.answer,
        "time": result.execution_time_seconds,
        "retries": result.retries,
        "weights_used": weight_used,
        "errors": result.errors,
        "success": result.success,
        "code_executed": result.code_executed,
        "raw_statistics": result.raw_statistics,
        "cached": False,
    })

@app.route("/api/analyze/stream", methods=["POST"])
def api_analyze_stream():
    """SSE endpoint: streams status updates then the final result."""
    data = request.json
    question = data.get("question", "").strip()
    dataset = data.get("dataset", "")
    agent_type = data.get("agent", "single")
    model = data.get("model", "gpt-4o")
    provider = data.get("provider")
    temperature = data.get("temperature", 0.2)

    if not question or not dataset:
        return jsonify({"error": "question and dataset required"}), 400

    # Check cache first
    cache_key = _web_cache_key(agent_type, model, dataset, question)
    cached = _load_web_cache(cache_key)
    if cached is not None:
        ds_info = get_dataset_info(dataset)
        weight_used = check_weight_usage(cached.code_executed, ds_info.weight_column)

        def generate_cached():
            yield f"data: {json.dumps({'type': 'status', 'message': 'Loading from cache...'})}\n\n"
            yield f"data: {json.dumps({'type': 'result', 'data': {'answer': cached.answer, 'time': cached.execution_time_seconds, 'retries': cached.retries, 'weights_used': weight_used, 'errors': cached.errors, 'success': cached.success, 'raw_statistics': cached.raw_statistics, 'cached': True}})}\n\n"

        return Response(generate_cached(), mimetype="text/event-stream")

    q = queue.Queue()

    def run_agent():
        try:
            agent_label = {"single": "Single Agent", "multi": "Multi-Agent", "rag": "RAG Agent"}.get(agent_type, agent_type)
            q.put({"type": "status", "message": f"Initializing {agent_label}..."})
            agent = AGENT_CLASSES[agent_type](model=model, provider=provider, temperature=temperature)

            q.put({"type": "status", "message": f"Analyzing with {model}..."})
            result = agent.analyze(question, dataset)

            # Save to cache
            _save_web_cache(cache_key, result)

            ds_info = get_dataset_info(dataset)
            weight_used = check_weight_usage(result.code_executed, ds_info.weight_column)

            q.put({"type": "result", "data": {
                "answer": result.answer,
                "time": result.execution_time_seconds,
                "retries": result.retries,
                "weights_used": weight_used,
                "errors": result.errors,
                "success": result.success,
                "raw_statistics": result.raw_statistics,
                "cached": False,
            }})
        except Exception as e:
            q.put({"type": "error", "message": str(e)})

    thread = Thread(target=run_agent, daemon=True)
    thread.start()

    def generate():
        while True:
            try:
                msg = q.get(timeout=120)
                yield f"data: {json.dumps(msg)}\n\n"
                if msg["type"] in ("result", "error"):
                    break
            except queue.Empty:
                yield f"data: {json.dumps({'type': 'status', 'message': 'Still processing...'})}\n\n"

    return Response(generate(), mimetype="text/event-stream")


@app.route("/api/compare", methods=["POST"])
def api_compare():
    data = request.json
    question = data.get("question", "").strip()
    dataset = data.get("dataset", "")
    agents = data.get("agents", ["single", "multi", "rag"])
    model = data.get("model", "gpt-4o")
    provider = data.get("provider")
    temperature = data.get("temperature", 0.2)
    question_id = data.get("question_id")

    if not question or not dataset:
        return jsonify({"error": "question and dataset required"}), 400

    ds_info = get_dataset_info(dataset)
    results_list = []

    for agent_type in agents:
        if agent_type not in AGENT_CLASSES:
            continue
        try:
            # Check cache
            cache_key = _web_cache_key(agent_type, model, dataset, question)
            cached = _load_web_cache(cache_key)
            if cached is not None:
                result = cached
            else:
                agent = AGENT_CLASSES[agent_type](model=model, provider=provider, temperature=temperature)
                result = agent.analyze(question, dataset)
                _save_web_cache(cache_key, result)

            weight_used = check_weight_usage(result.code_executed, ds_info.weight_column)

            entry = {
                "agent": agent_type,
                "answer": result.answer,
                "time": result.execution_time_seconds,
                "retries": result.retries,
                "weights_used": weight_used,
                "errors": result.errors,
                "success": result.success,
                "cached": cached is not None,
            }

            # Score against ground truth if question_id provided
            if question_id:
                all_questions = load_questions()
                q_def = next((q for q in all_questions if q["id"] == question_id), None)
                if q_def and q_def.get("ground_truth_key"):
                    try:
                        gt = compute_ground_truth(q_def["ground_truth_key"])
                        score = evaluate_answer(result, q_def, gt, agent_type)
                        entry["accuracy"] = round(score.accuracy, 3)
                        entry["completeness"] = round(score.completeness, 3)
                    except Exception:
                        pass

            results_list.append(entry)
        except Exception as e:
            results_list.append({
                "agent": agent_type,
                "answer": "",
                "time": 0,
                "errors": [str(e)],
                "success": False,
            })

    return jsonify({"results": results_list})

@app.route("/api/results/list")
def api_results_list():
    results_dir = Path("experiments/results")
    if not results_dir.exists():
        return jsonify([])
    files = sorted(results_dir.glob("*.json"), reverse=True)
    return jsonify([f.name for f in files])

@app.route("/api/results/<filename>")
def api_results_file(filename):
    path = Path("experiments/results") / filename
    if not path.exists() or not path.suffix == ".json":
        return jsonify({"error": "File not found"}), 404
    with open(path) as f:
        return jsonify(json.load(f))

@app.route("/api/preset-questions/<dataset>")
def api_preset_questions(dataset):
    all_questions = load_questions()
    dataset_questions = [q for q in all_questions if q["dataset"] == dataset]
    return jsonify(dataset_questions)


@app.route("/api/score", methods=["POST"])
def api_score():
    """Score a single answer against ground truth."""
    data = request.json
    question_id = data.get("question_id")
    answer_text = data.get("answer", "")
    agent_type = data.get("agent", "single")

    if not question_id:
        return jsonify({"error": "question_id required"}), 400

    all_questions = load_questions()
    q_def = next((q for q in all_questions if q["id"] == question_id), None)
    if not q_def:
        return jsonify({"error": f"Unknown question_id: {question_id}"}), 404

    gt_key = q_def.get("ground_truth_key")
    if not gt_key:
        return jsonify({"error": "No ground truth available for this question"}), 400

    try:
        gt = compute_ground_truth(gt_key)
        # Build a minimal AnalysisResult for scoring
        result = AnalysisResult(
            question=q_def["question"],
            dataset=q_def["dataset"],
            answer=answer_text,
        )
        score = evaluate_answer(result, q_def, gt, agent_type)
        return jsonify({
            "accuracy": round(score.accuracy, 3),
            "completeness": round(score.completeness, 3),
            "ground_truth": gt,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/error-analysis")
def api_error_analysis():
    """Aggregate error analysis from cached experiment results."""
    results_dir = Path("experiments/results")
    if not results_dir.exists():
        return jsonify({"error": "No experiment results found"}), 404

    files = sorted(results_dir.glob("*.json"), reverse=True)
    if not files:
        return jsonify({"error": "No result files found"}), 404

    # Use the most recent result file, or a specific one if requested
    filename = request.args.get("file")
    if filename:
        path = results_dir / filename
        if not path.exists():
            return jsonify({"error": "File not found"}), 404
    else:
        path = files[0]

    with open(path) as f:
        data = json.load(f)

    # Result files may be a list of scores directly or {"scores": [...]}
    if isinstance(data, list):
        scores = data
    else:
        scores = data.get("scores", [])
    errors_by_agent = {}
    errors_by_question_type = {}
    errors_by_dataset = {}
    weight_failures = []
    low_accuracy = []

    for s in scores:
        agent = s.get("agent", "unknown")
        qid = s.get("question_id", "")
        dataset = s.get("dataset", "")
        accuracy = s.get("accuracy", 0)
        had_error = s.get("had_error", False)
        weight_used = s.get("weight_used", False)
        details = s.get("details", {})
        q_type = details.get("ground_truth", {}).get("type", details.get("type", "unknown"))

        # Count errors by agent
        if agent not in errors_by_agent:
            errors_by_agent[agent] = {"total": 0, "errors": 0, "low_accuracy": 0, "weight_failures": 0}
        errors_by_agent[agent]["total"] += 1
        if had_error:
            errors_by_agent[agent]["errors"] += 1
        if accuracy < 0.5:
            errors_by_agent[agent]["low_accuracy"] += 1
        if not weight_used:
            errors_by_agent[agent]["weight_failures"] += 1

        # Count by question type
        if q_type not in errors_by_question_type:
            errors_by_question_type[q_type] = {"total": 0, "errors": 0, "avg_accuracy": []}
        errors_by_question_type[q_type]["total"] += 1
        if had_error or accuracy < 0.5:
            errors_by_question_type[q_type]["errors"] += 1
        errors_by_question_type[q_type]["avg_accuracy"].append(accuracy)

        # Count by dataset
        if dataset not in errors_by_dataset:
            errors_by_dataset[dataset] = {"total": 0, "errors": 0, "avg_accuracy": []}
        errors_by_dataset[dataset]["total"] += 1
        if had_error or accuracy < 0.5:
            errors_by_dataset[dataset]["errors"] += 1
        errors_by_dataset[dataset]["avg_accuracy"].append(accuracy)

        # Track specific failures
        if not weight_used:
            weight_failures.append({"agent": agent, "question_id": qid, "dataset": dataset})
        if accuracy < 0.5:
            low_accuracy.append({"agent": agent, "question_id": qid, "dataset": dataset, "accuracy": accuracy, "type": q_type})

    # Compute averages
    for v in errors_by_question_type.values():
        accs = v.pop("avg_accuracy")
        v["avg_accuracy"] = round(sum(accs) / len(accs), 3) if accs else 0
    for v in errors_by_dataset.values():
        accs = v.pop("avg_accuracy")
        v["avg_accuracy"] = round(sum(accs) / len(accs), 3) if accs else 0

    return jsonify({
        "file": path.name,
        "total_scores": len(scores),
        "errors_by_agent": errors_by_agent,
        "errors_by_question_type": errors_by_question_type,
        "errors_by_dataset": errors_by_dataset,
        "weight_failures": weight_failures,
        "low_accuracy_questions": sorted(low_accuracy, key=lambda x: x["accuracy"]),
        "all_scores": scores,
    })


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5001, debug=True, allow_unsafe_werkzeug=True)
