"""
Simple Flask server exposing a /chat endpoint and a minimal UI.

Run:
  python -m src.chatbot.server --model models/intent_model_v1.joblib --intents data/intents.json
"""
import os
import sys
import argparse
import json
from typing import Any, Dict

# Ensure package importable
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from flask import Flask, request, jsonify, render_template_string
from chatbot.data_loader import load_intents
from chatbot.model import IntentModel
from chatbot.response_selector import ResponseSelector

app = Flask(__name__)
INTENTS = {}
MODEL = None
SELECTOR = None

SIMPLE_UI = """
<!doctype html>
<title>MedAssist Chat</title>
<h1>MedAssist Chat UI</h1>
<div id="chat" style="max-width:800px;">
</div>
<input id="msg" style="width:80%;" placeholder="Type your message...">
<button onclick="send()">Send</button>
<script>
async function send(){
  const m = document.getElementById('msg').value;
  if(!m) return;
  const r = await fetch('/chat', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({message: m})
  });
  const j = await r.json();
  const div = document.getElementById('chat');
  div.innerHTML += "<p><b>You:</b> "+m+"</p>";
  div.innerHTML += "<p><b>Bot:</b> "+j.response+"</p><hr>";
  document.getElementById('msg').value = '';
}
</script>
"""

@app.route("/")
def index():
    return render_template_string(SIMPLE_UI)


@app.route("/chat", methods=["POST"])
def chat() -> Any:
    payload = request.get_json(silent=True)
    if not payload or "message" not in payload:
        return jsonify({"error": "Please provide JSON with a 'message' field."}), 400
    text = payload["message"]
    try:
        preds, conf = MODEL.predict(text, top_k=3)
        best_tag, best_prob = preds[0]
        # thresholding
        threshold = float(request.args.get("threshold", 0.45))
        if best_prob < threshold:
            return jsonify({
                "intent": None,
                "confidence": best_prob,
                "response": "I'm not confident about your request. Could you provide more details or rephrase?"
            })
        response = SELECTOR.select(best_tag)
        safety = SELECTOR.get_safety_note(best_tag)
        if safety:
            response = f"{response}\n\nSafety note: {safety}"
        escalation = SELECTOR.get_escalation(best_tag)
        if escalation:
            for k in escalation.get("keywords", []):
                if k in text.lower():
                    response = f"{escalation.get('action')}\n\n{response}"
                    break
        return jsonify({
            "intent": best_tag,
            "confidence": best_prob,
            "response": response
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def load_globals(model_path: str, intents_path: str):
    global INTENTS, MODEL, SELECTOR
    INTENTS = load_intents(intents_path)
    MODEL = IntentModel()
    MODEL.load(model_path)
    SELECTOR = ResponseSelector(INTENTS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/intent_model_v1.joblib")
    parser.add_argument("--intents", type=str, default="data/intents.json")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print("Model not found. Train first:", args.model)
        return

    load_globals(args.model, args.intents)
    print("Starting server on http://%s:%d" % (args.host, args.port))
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
