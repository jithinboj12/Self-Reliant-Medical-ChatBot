import re
from typing import List
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure required NLTK data is available. This will download once per machine if missing.
try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("corpora/wordnet")
    nltk.data.find("corpora/omw-1.4")
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("punkt")
    nltk.download("wordnet")
    nltk.download("omw-1.4")
    nltk.download("stopwords")

_LEMMATIZER = WordNetLemmatizer()
_STOPWORDS = set(stopwords.words("english"))


class Preprocessor:
    def __init__(self, custom_stopwords: List[str] = None):
        if custom_stopwords:
            self.stopwords = _STOPWORDS.union(set(custom_stopwords))
        else:
            self.stopwords = _STOPWORDS

    def normalize(self, text: str) -> str:
        # Lowercase and remove non-alphanumerics (keep spaces)
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text: str) -> List[str]:
        """
        Full tokenize pipeline that returns a list of normalized tokens.
        This function is compatible with sklearn's TfidfVectorizer tokenizer parameter.
        """
        if not isinstance(text, str):
            text = str(text)
        normalized = self.normalize(text)
        tokens = nltk.word_tokenize(normalized)
        lemmas = []
        for t in tokens:
            if t in self.stopwords:
                continue
            lemma = _LEMMATIZER.lemmatize(t)
            if lemma:
                lemmas.append(lemma)
        return lemmas
