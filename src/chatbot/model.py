import os
from typing import List, Tuple, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

from .preprocessor import Preprocessor


class IntentModel:
    def __init__(self):
        self.preprocessor = Preprocessor()
        # We will create vectorizer and classifier during training or load
        self.vectorizer = None
        self.classifier = None
        self.le = LabelEncoder()
        self.pipeline = None

    def train(self, texts: List[str], labels: List[str], test_size=0.15, random_state=42) -> Dict[str, Any]:
        """
        Train the TF-IDF + LR pipeline and return evaluation metrics.
        """
        # Label encoding
        y = self.le.fit_transform(labels)

        # Build pipeline using our preprocessor's tokenizer
        # We set lowercase=False and token_pattern=None to rely on custom tokenizer
        self.vectorizer = TfidfVectorizer(
            tokenizer=self.preprocessor.tokenize,
            lowercase=False,
            token_pattern=None,
            ngram_range=(1, 2),
            max_features=20000
        )
        self.classifier = LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=200)

        # Fit vectorizer and classifier
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, y)

        # Save pipeline for convenience
        self.pipeline = Pipeline([
            ("vectorizer", self.vectorizer),
            ("classifier", self.classifier)
        ])

        # Evaluate
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        y_pred = self.classifier.predict(X_test)
        report = classification_report(y_test, y_pred, target_names=self.le.classes_, output_dict=True)
        return report

    def predict(self, text: str, top_k: int = 1) -> Tuple[List[Tuple[str, float]], float]:
        """
        Predict top_k intents for a single text.
        Returns list of (label, probability) and the highest probability (confidence).
        """
        if self.vectorizer is None or self.classifier is None:
            raise ValueError("Model not loaded or trained.")

        X = self.vectorizer.transform([text])
        # use predict_proba if available
        if hasattr(self.classifier, "predict_proba"):
            probs = self.classifier.predict_proba(X)[0]
            # get top_k indices
            idx_sorted = probs.argsort()[::-1][:top_k]
            results = []
            for i in idx_sorted:
                label = self.le.inverse_transform([i])[0]
                results.append((label, float(probs[i])))
            confidence = float(probs.max())
            return results, confidence
        else:
            pred = self.classifier.predict(X)[0]
            label = self.le.inverse_transform([pred])[0]
            return [(label, 1.0)], 1.0

    def save(self, path: str):
        """
        Save vectorizer, classifier, label encoder in a single joblib file.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "vectorizer": self.vectorizer,
            "classifier": self.classifier,
            "label_encoder": self.le
        }
        joblib.dump(data, path)

    def load(self, path: str):
        """
        Load saved joblib model file.
        """
        data = joblib.load(path)
        self.vectorizer = data["vectorizer"]
        self.classifier = data["classifier"]
        self.le = data["label_encoder"]
        self.pipeline = Pipeline([
            ("vectorizer", self.vectorizer),
            ("classifier", self.classifier)
        ])
