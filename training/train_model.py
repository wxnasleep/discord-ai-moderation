"""Train a compact Russian toxicity score model from a local CSV dataset.

CSV columns: text,score. Scores range from 1 (benign) to 10 (severe).
Keep the source data private unless you have permission to publish it.
"""

import csv
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_union import FeatureUnion
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "training.csv"
DEFAULT_OUTPUT = ROOT / "models" / "toxicity_model.joblib"


def read_examples(path: Path) -> tuple[list[str], list[float]]:
    texts: list[str] = []
    scores: list[float] = []
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not {"text", "score"}.issubset(reader.fieldnames or []):
            raise ValueError("CSV must have 'text' and 'score' columns.")
        for line_number, row in enumerate(reader, start=2):
            text = (row.get("text") or "").strip()
            try:
                score = float(row.get("score", ""))
            except ValueError as error:
                raise ValueError(f"Invalid score on CSV line {line_number}.") from error
            if not text or not np.isfinite(score) or not 1 <= score <= 10:
                raise ValueError(f"Invalid text or score on CSV line {line_number}; scores must be 1..10.")
            texts.append(text[:4000])
            scores.append(score)
    if len(texts) < 20:
        raise ValueError("Add at least 20 labelled examples before training.")
    return texts, scores


def build_model() -> Pipeline:
    features = FeatureUnion(
        [
            ("char", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=1, max_features=100_000)),
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, max_features=50_000)),
        ]
    )
    return Pipeline([("features", features), ("regressor", Ridge(alpha=2.0))])


def main() -> None:
    load_dotenv(ROOT / ".env")
    dataset_path = Path(os.getenv("TRAINING_DATA_PATH", str(DEFAULT_DATASET)))
    output_path = Path(os.getenv("MODEL_OUTPUT_PATH", str(DEFAULT_OUTPUT)))
    texts, scores = read_examples(dataset_path)

    model = build_model()
    model.fit(texts, scores)

    artifact = {
        "score_model": model,
        "feedback_texts": [],
        "feedback_scores": [],
        "model_version": 1,
        "training_examples": len(texts),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    joblib.dump(artifact, temporary_path, compress=3)
    temporary_path.replace(output_path)
    print(f"Saved model from {len(texts)} examples: {output_path}")


if __name__ == "__main__":
    main()
