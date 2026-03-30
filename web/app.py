"""Flask backend for the LLM Data Scientist dashboard."""

import json
import sys
import time
import queue
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

    q = queue.Queue()

    def run_agent():
        try:
            agent_label = {"single": "Single Agent", "multi": "Multi-Agent", "rag": "RAG Agent"}.get(agent_type, agent_type)
            q.put({"type": "status", "message": f"Initializing {agent_label}..."})
            agent = AGENT_CLASSES[agent_type](model=model, provider=provider, temperature=temperature)

            q.put({"type": "status", "message": f"Analyzing with {model}..."})
            result = agent.analyze(question, dataset)

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
            agent = AGENT_CLASSES[agent_type](model=model, provider=provider, temperature=temperature)
            result = agent.analyze(question, dataset)
            weight_used = check_weight_usage(result.code_executed, ds_info.weight_column)

            entry = {
                "agent": agent_type,
                "answer": result.answer,
                "time": result.execution_time_seconds,
                "retries": result.retries,
                "weights_used": weight_used,
                "errors": result.errors,
                "success": result.success,
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


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5001, debug=True, allow_unsafe_werkzeug=True)
