import random
from typing import Dict, List


class ResponseSelector:
    def __init__(self, intents_json: Dict):
        """
        intents_json: loaded JSON dict of intents file
        """
        self.map = {}
        for item in intents_json.get("intents", []):
            tag = item.get("tag")
            responses = item.get("responses", [])
            self.map[tag] = {
                "responses": responses,
                "escalation": item.get("escalation"),
                "safety_note": item.get("safety_note")
            }

    def select(self, intent_tag: str) -> str:
        info = self.map.get(intent_tag)
        if not info:
            return "Sorry, I don't have an answer for that."
        responses = info.get("responses", [])
        if not responses:
            return "I don't have a prepared response for that intent."
        return random.choice(responses)

    def get_escalation(self, intent_tag: str):
        info = self.map.get(intent_tag, {})
        return info.get("escalation")

    def get_safety_note(self, intent_tag: str):
        info = self.map.get(intent_tag, {})
        return info.get("safety_note")
