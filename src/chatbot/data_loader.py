import json
from typing import List, Dict


def load_intents(path: str) -> Dict:
    """
    Load the intents JSON file.
    Returns the parsed JSON as a dict.
    """
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data


def build_training_examples(intents_json: Dict):
    """
    Convert the intents JSON into (texts, labels) lists for training.
    """
    texts = []
    labels = []
    intents = intents_json.get("intents", [])
    for item in intents:
        tag = item.get("tag")
        patterns = item.get("patterns", [])
        for p in patterns:
            texts.append(p)
            labels.append(tag)
    return texts, labels
