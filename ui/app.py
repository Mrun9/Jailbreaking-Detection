"""
ui/app.py
=========
Placeholder Flask server for the jailbreak detection interface.
Full implementation coming in Week 4.

Run:
    python ui/app.py
Then open http://localhost:5000
"""

from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Jailbreak Detector</title>
    <style>
        body { font-family: sans-serif; max-width: 700px; margin: 60px auto; padding: 0 20px; }
        textarea { width: 100%; height: 120px; font-size: 14px; padding: 8px; }
        button { margin-top: 10px; padding: 10px 24px; background: #e74c3c;
                 color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 14px; }
        #result { margin-top: 20px; padding: 14px; border-radius: 6px; font-size: 15px; }
        .jailbreak { background: #fdecea; border-left: 4px solid #e74c3c; }
        .benign    { background: #eafaf1; border-left: 4px solid #27ae60; }
    </style>
</head>
<body>
    <h2>Jailbreak Prompt Detector</h2>
    <p>Enter a prompt below to check whether it is a jailbreak attempt.</p>
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
    div.className = data.is_jailbreak ? 'jailbreak' : 'benign';
    div.innerHTML = data.is_jailbreak
        ? `<strong>Jailbreak detected</strong> (stage: ${data.stage}, confidence: ${(data.confidence*100).toFixed(1)}%)`
        : `<strong>Benign prompt</strong> (stage: ${data.stage}, confidence: ${(data.confidence*100).toFixed(1)}%)`;
}
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/detect', methods=['POST'])
def detect():
    """
    Placeholder endpoint — returns a dummy response.
    Will be wired to detector.py in Week 4.
    """
    data = request.get_json()
    prompt = data.get('prompt', '')

    # Placeholder logic — replace with: detector.detect(prompt)
    dummy_result = {
        "prompt": prompt,
        "is_jailbreak": False,
        "stage": "placeholder",
        "confidence": 0.0,
        "latency_ms": 0.0
    }
    return jsonify(dummy_result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
