# Text Similarity Checker

A small full-stack NLP project: a **FastAPI** backend with a **vanilla JavaScript** frontend. Compare two texts, or rank a list of documents against a query, using either lexical or semantic similarity.

## Features

- **Compare** two texts and get a similarity score from 0 to 100%.
- **Rank** up to 100 documents against a query, best match first.
- **Two methods**
  - **TF-IDF** (default): lexical similarity based on word and bigram overlap. Fast, no extra downloads.
  - **Semantic**: sentence embeddings (`all-MiniLM-L6-v2`) that capture meaning, so "car" and "automobile" count as similar.
- Plain-language labels for every score.
- Responsive UI with automatic dark mode.
- Interactive API docs at `/docs` (Swagger UI).

## Screenshot

<!-- Add a screenshot here, e.g. ![App screenshot](docs/screenshot.png) -->

## Quick start

```bash
git clone <your-repo-url>
cd similarity-checker

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000 for the app, or http://127.0.0.1:8000/docs for the API docs.

### Optional: semantic mode

```bash
pip install sentence-transformers
```

The first semantic request downloads the model (about 80 MB). Without the package installed, semantic requests return a `400` with an install hint.

## How scoring works

Both methods produce a cosine similarity between 0 and 1, which is mapped to a label:

| Score | Label |
|-------|-------|
| 0.85 and above | very similar |
| 0.60 to 0.85 | similar |
| 0.30 to 0.60 | somewhat related |
| below 0.30 | different |

TF-IDF lowercases the text, strips punctuation, removes English stop words, and compares unigrams and bigrams. Semantic mode encodes each text into an embedding vector and compares the vectors.

## API

### `POST /api/compare`

```bash
curl -X POST http://127.0.0.1:8000/api/compare \
  -H "Content-Type: application/json" \
  -d '{"text1": "Python programming tutorial for beginners", "text2": "A beginner tutorial on Python programming", "method": "tfidf"}'
```

```json
{ "score": 0.403, "label": "somewhat related", "method": "tfidf" }
```

### `POST /api/rank`

```bash
curl -X POST http://127.0.0.1:8000/api/rank \
  -H "Content-Type: application/json" \
  -d '{"query": "learn python programming", "documents": ["Pizza recipes", "Python programming tutorial for beginners"], "method": "tfidf"}'
```

```json
{
  "method": "tfidf",
  "results": [
    { "index": 1, "text": "Python programming tutorial for beginners", "score": 0.3749, "label": "somewhat related" },
    { "index": 0, "text": "Pizza recipes", "score": 0.0, "label": "different" }
  ]
}
```

`method` is optional and defaults to `tfidf`. Empty text returns `422`; semantic mode without `sentence-transformers` returns `400`.

## Project structure

```
similarity-checker/
├── app/
│   ├── main.py          # FastAPI routes and request validation
│   └── similarity.py    # TF-IDF and semantic scoring logic
├── static/
│   └── index.html       # Frontend (HTML, CSS, JS in one file)
├── tests/
│   └── test_api.py      # pytest API tests
├── requirements.txt
└── .gitignore
```

## Tests

```bash
pytest
```

## Ideas to extend

- Upload `.txt` or `.pdf` files instead of pasting text
- Highlight the overlapping words or sentences
- Store comparison history in PostgreSQL
- Dockerize and deploy
