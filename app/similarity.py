"""Core NLP logic: TF-IDF (lexical) and sentence-embedding (semantic) similarity."""
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_model = None


def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tfidf_scores(query: str, docs: list[str]) -> list[float]:
    """Similarity of `query` against each doc using TF-IDF + cosine."""
    texts = [preprocess(query)] + [preprocess(d) for d in docs]
    matrix = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform(texts)
    return cosine_similarity(matrix[0], matrix[1:])[0].tolist()


def semantic_scores(query: str, docs: list[str]) -> list[float]:
    """Similarity using sentence embeddings (needs sentence-transformers)."""
    global _model
    try:
        from sentence_transformers import SentenceTransformer, util
    except ImportError as e:
        raise RuntimeError(
            "Semantic method needs: pip install sentence-transformers"
        ) from e
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    q = _model.encode(query, convert_to_tensor=True)
    d = _model.encode(docs, convert_to_tensor=True)
    return util.cos_sim(q, d)[0].tolist()


def compute(query: str, docs: list[str], method: str = "tfidf") -> list[float]:
    scores = semantic_scores(query, docs) if method == "semantic" else tfidf_scores(query, docs)
    return [max(0.0, min(1.0, round(s, 4))) for s in scores]


def label(score: float) -> str:
    if score >= 0.85:
        return "very similar"
    if score >= 0.6:
        return "similar"
    if score >= 0.3:
        return "somewhat related"
    return "different"
