# Self-Reliant Medical Chatbot Using Machine Learning for Smart Intent Classification

A cross-platform, Python-based medical chatbot focused on intent classification using a lightweight, explainable ML pipeline. This repository contains a complete, runnable project with training, model persistence, a CLI for testing, and a simple Flask API. The system is intended as a general-purpose medical assistant prototype (triage, appointment, medication guidance, first-aid, preventive advice, mental health guidance, and escalation to professionals). It is NOT a replacement for professional medical advice — include appropriate disclaimers when deploying.

Key features
- NLTK-based preprocessing (tokenization, lemmatization, stopword removal)
- TF-IDF + Logistic Regression intent classifier
- Configurable confidence threshold and fallback / escalation
- Persisted model (joblib) and vectorizer
- CLI and Flask HTTP API for integration
- Example multi-intent dataset for medical uses
- Easy to extend with new intents and patterns

Quickstart (Linux/macOS/Windows)
1. Clone repository
2. Create and activate a virtual environment (recommended)
   - python -m venv .venv
   - source .venv/bin/activate  (Windows: .venv\Scripts\activate)
3. Install dependencies
   - pip install -r requirements.txt
4. Train model:
   - python scripts/train.py --intents data/intents.json --model-path models/intent_model_v1.joblib
   This will create `models/intent_model_v1.joblib`.
5. Run CLI:
   - python -m src.chatbot.cli --model models/intent_model_v1.joblib --intents data/intents.json
6. Run API server:
   - python -m src.chatbot.server --model models/intent_model_v1.joblib --intents data/intents.json
   - Open http://127.0.0.1:5000/ui (simple chat UI) or POST JSON to /chat

Safety & medical disclaimer
This project is a research/prototype tool. Do not deploy as a sole source of medical advice. Always include a disclaimer and provide escalation to clinical professionals or emergency services for critical cases.

Extending the system
- Add intents/patterns in data/intents.json
- Re-train to include new data
- Swap model components: try larger models, embedding-based classifiers, or integrate entity extraction
