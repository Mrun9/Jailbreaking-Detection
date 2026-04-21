"""
ui/app.py
=========
Flask UI for the jailbreak detector.

Run:
    python ui/app.py
Then open http://localhost:5000
"""

import csv
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
DATASET_CSV = REPO_ROOT / "results" / "collected_prompts.csv"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from detector import load_project_detector


app = Flask(__name__)


def _normalize_label(value) -> int:
    """Normalize common 0/1 and boolean-style labels."""
    if value is None:
        raise ValueError("Missing jailbreak label.")

    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "jailbreak", "jb"}:
        return 1
    if text in {"0", "false", "no", "n", "benign", "safe"}:
        return 0
    raise ValueError(f"Unsupported jailbreak label: {value}")


def _load_seed_jailbreak_prompts() -> list[str]:
    """Load known jailbreak prompts from the project CSV for cache seeding."""
    if not DATASET_CSV.exists():
        print(f"[ui] Cache seed CSV not found: {DATASET_CSV}")
        return []

    prompts = []
    with DATASET_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            prompt = str(row.get("prompt", "")).strip()
            if not prompt:
                continue
            try:
                is_jailbreak = _normalize_label(row.get("jailbreak"))
            except ValueError:
                continue
            if is_jailbreak == 1:
                prompts.append(prompt)
    print(f"[ui] Loaded {len(prompts)} jailbreak prompts for cache seeding.")
    return prompts


def _create_detector():
    """Initialize the detector and seed its in-memory cache on startup."""
    detector = load_project_detector(auto_update_cache=True)
    jailbreak_prompts = _load_seed_jailbreak_prompts()
    if jailbreak_prompts:
        detector.seed_cache(jailbreak_prompts)
    else:
        print("[ui] Detector started without a seeded cache.")
    return detector


DETECTOR = _create_detector()


HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Jailbreak Detector</title>
    <style>
        body {
            font-family: sans-serif;
            max-width: 760px;
            margin: 60px auto;
            padding: 0 20px;
        }
        textarea {
            width: 100%;
            height: 140px;
            font-size: 14px;
            padding: 10px;
        }
        button {
            margin-top: 12px;
            padding: 10px 24px;
            background: #e74c3c;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }
        #result {
            margin-top: 20px;
            padding: 16px;
            border-radius: 6px;
            font-size: 15px;
            line-height: 1.5;
        }
        .jailbreak { background: #fdecea; border-left: 4px solid #e74c3c; }
        .benign { background: #eafaf1; border-left: 4px solid #27ae60; }
        .unknown { background: #f5f5f5; border-left: 4px solid #6b7280; }
        .meta { margin-top: 8px; color: #333; }
        code { background: rgba(0,0,0,0.06); padding: 1px 4px; border-radius: 4px; }
    </style>
</head>
<body>
    <h2>Jailbreak Prompt Detector</h2>
    <p>Enter a prompt below to classify it and see whether the decision came from a cache hit or cache miss.</p>
    <textarea id="prompt" placeholder="Type your prompt here..."></textarea>
    <br>
    <button onclick="check()">Check Prompt</button>
    <div id="result"></div>

<script>
async function check() {
    const prompt = document.getElementById('prompt').value.trim();
    if (!prompt) return;

    const res = await fetch('/detect', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({prompt})
    });
    const data = await res.json();

    const div = document.getElementById('result');
    div.className = data.classification === 'jailbreak' ? 'jailbreak' :
                    data.classification === 'benign' ? 'benign' : 'unknown';

    const headline = data.classification === 'jailbreak'
        ? '<strong>Jailbreak attempt flagged</strong>'
        : data.classification === 'benign'
            ? '<strong>Prompt classified as benign</strong>'
            : '<strong>Classification unresolved</strong>';

    const matchedPrompt = data.matched_prompt
        ? `<div class="meta"><strong>Matched cached prompt:</strong> ${data.matched_prompt}</div>`
        : '';

    div.innerHTML = `
        ${headline}
        <div class="meta"><strong>Classification:</strong> ${data.classification}</div>
        <div class="meta"><strong>Confidence:</strong> ${(data.confidence * 100).toFixed(1)}%</div>
        <div class="meta"><strong>Decision time:</strong> ${data.time_taken_ms.toFixed(3)} ms</div>
        <div class="meta"><strong>Cache:</strong> ${data.cache_status}</div>
        <div class="meta"><strong>Detector stage:</strong> ${data.stage}</div>
        ${matchedPrompt}
    `;
}
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/detect", methods=["POST"])
def detect():
    """Run the cached detector against a single prompt."""
    data = request.get_json(silent=True) or {}
    prompt = str(data.get("prompt", "")).strip()

    if not prompt:
        return jsonify({"error": "Prompt is required."}), 400

    result = DETECTOR.detect(prompt)
    response = {
        "prompt": prompt,
        "flagged": result["is_jailbreak"],
        "classification": result["decision"],
        "confidence": result["confidence"],
        "time_taken_ms": result["latency_ms"],
        "cache_status": "hit" if result["stage"] == "cache" else "miss",
        "stage": result["stage"],
        "similarity": result["similarity"],
        "matched_prompt": result["matched_prompt"],
    }
    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
