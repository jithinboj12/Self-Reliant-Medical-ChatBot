#!/usr/bin/env python3
"""
Train the intent classification model and save it to disk.

Usage:
  python scripts/train.py --intents data/intents.json --model-path models/intent_model_v1.joblib
"""
import os
import argparse
import sys

# Make 'src' importable when running script directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from chatbot.data_loader import load_intents, build_training_examples
from chatbot.model import IntentModel
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--intents", type=str, default="data/intents.json")
    parser.add_argument("--model-path", type=str, default="models/intent_model_v1.joblib")
    args = parser.parse_args()

    print("Loading intents:", args.intents)
    intents = load_intents(args.intents)
    texts, labels = build_training_examples(intents)
    print(f"Prepared {len(texts)} training examples across {len(set(labels))} intents.")

    model = IntentModel()
    print("Training model...")
    report = model.train(texts, labels)
    print("Training complete. Evaluation summary:")
    # Print simple summary
    print(json.dumps(report, indent=2))

    print("Saving model to", args.model_path)
    os.makedirs(os.path.dirname(args.model_path) or ".", exist_ok=True)
    model.save(args.model_path)
    print("Done.")


if __name__ == "__main__":
    main()
