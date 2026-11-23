from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_identical_texts_score_high():
    r = client.post("/api/compare", json={"text1": "I love Python programming", "text2": "I love Python programming"})
    assert r.status_code == 200
    assert r.json()["score"] > 0.95


def test_unrelated_texts_score_low():
    r = client.post("/api/compare", json={"text1": "Cats sleep all day long", "text2": "Stock markets fell sharply today"})
    assert r.json()["score"] < 0.2


def test_rank_orders_best_first():
    r = client.post("/api/rank", json={
        "query": "learn python programming",
        "documents": ["Best pizza recipes", "Python programming tutorial for beginners", "Gardening tips"],
    })
    assert r.json()["results"][0]["index"] == 1


def test_empty_text_rejected():
    r = client.post("/api/compare", json={"text1": "", "text2": "hi"})
    assert r.status_code == 422


def test_semantic_without_dependency_returns_400(monkeypatch):
    import sys

    # Simulate sentence-transformers not being installed.
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    r = client.post("/api/compare", json={"text1": "hello", "text2": "world", "method": "semantic"})
    assert r.status_code == 400
    assert "sentence-transformers" in r.json()["detail"]
