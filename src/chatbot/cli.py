import argparse
import os
import sys
import json

# Ensure package imports work when executed as module from repo root
# If running `python -m src.chatbot.cli` from repo root, this is unnecessary.
# But keep a fallback for direct execution.
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from chatbot.data_loader import load_intents
from chatbot.model import IntentModel
from chatbot.response_selector import ResponseSelector


def interactive_loop(model: IntentModel, selector: ResponseSelector, threshold: float = 0.45):
    print("Medical chatbot CLI. Type 'exit' or 'quit' to stop.")
    while True:
        text = input("You: ").strip()
        if text.lower() in ("exit", "quit"):
            print("Goodbye.")
            break
        try:
            preds, conf = model.predict(text, top_k=3)
            best_tag, best_prob = preds[0]
            if best_prob < threshold:
                print("Bot: I'm not confident about that. Could you rephrase or provide more details?")
                continue
            response = selector.select(best_tag)
            safety = selector.get_safety_note(best_tag)
            if safety:
                response = f"{response}\n\nSafety note: {safety}"
            escalation = selector.get_escalation(best_tag)
            if escalation:
                # check if escalation keywords appear (simple check)
                keys = escalation.get("keywords", [])
                for k in keys:
                    if k in text.lower():
                        response = f"{escalation.get('action')}\n\n{response}"
                        break
            print(f"Bot ({best_tag}, conf={best_prob:.2f}): {response}")
        except Exception as e:
            print("Bot: Sorry, an error occurred:", e)


def load_model(path: str) -> IntentModel:
    m = IntentModel()
    m.load(path)
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/intent_model_v1.joblib")
    parser.add_argument("--intents", type=str, default="data/intents.json")
    parser.add_argument("--threshold", type=float, default=0.45)
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print("Model not found. Train first:", args.model)
        return
    intents = load_intents(args.intents)
    model = load_model(args.model)
    selector = ResponseSelector(intents)
    interactive_loop(model, selector, threshold=args.threshold)


if __name__ == "__main__":
    main()
