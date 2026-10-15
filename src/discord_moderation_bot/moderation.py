"""Model loading and toxicity scoring."""

import asyncio
import logging
import math
import re
import unicodedata
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import normalize as l2_normalize


logger = logging.getLogger(__name__)


class ToxicityModerator:
    """Wrap the trained score model and keep inference off the event loop."""

    def __init__(self, model: dict[str, object]):
        if not isinstance(model, dict) or model.get("score_model") is None:
            raise ValueError("Model file must contain a 'score_model' entry.")

        self.score_model = model["score_model"]
        self.feedback_texts: list[str] = []
        feedback_scores: list[float] = []
        self.exact_feedback: dict[str, float] = {}

        for text, score in zip(model.get("feedback_texts", []), model.get("feedback_scores", [])):
            normalized = self.normalize(text)
            try:
                value = float(score)
            except (TypeError, ValueError):
                continue
            if normalized and math.isfinite(value) and 1 <= value <= 10:
                self.exact_feedback[normalized] = value

        self.feedback_texts = list(self.exact_feedback)
        self.feedback_scores = np.asarray(list(self.exact_feedback.values()), dtype=float)
        self.feedback_vectorizer = getattr(self.score_model, "named_steps", {}).get("features")
        self.feedback_matrix = None
        if self.feedback_vectorizer is not None and self.feedback_texts:
            self.feedback_matrix = l2_normalize(
                self.feedback_vectorizer.transform(self.feedback_texts), copy=True
            )
        self._semaphore = asyncio.Semaphore(1)

    @staticmethod
    def normalize(text: object) -> str:
        value = unicodedata.normalize("NFKC", str(text or "")).lower().replace("ё", "е")
        value = re.sub(r"<[@#][!&]?\d+>|https?://\S+", " ", value)
        return re.sub(r"\s+", " ", value).strip()[:4000]

    @classmethod
    def load(cls, path: Path) -> "ToxicityModerator":
        if not path.is_file():
            raise FileNotFoundError(
                f"Model not found: {path}. Train one with `python -m training.train_model`."
            )
        # joblib uses pickle internally: only load model files from sources you trust.
        return cls(joblib.load(path))

    def predict_score(self, text: str) -> float:
        normalized = self.normalize(text)
        if not normalized or re.fullmatch(r"[\W\d_]+|(?:[ах]+|(?:ha|he|lol)+)[!?., ]*", normalized):
            return 1.0

        exact = self.exact_feedback.get(normalized)
        if exact is not None:
            return exact

        base_value = float(self.score_model.predict([normalized])[0])
        base = float(np.clip(base_value, 1, 10)) if math.isfinite(base_value) else 1.0
        if self.feedback_matrix is None:
            return base

        vector = l2_normalize(self.feedback_vectorizer.transform([normalized]), copy=True)
        similarities = np.clip((self.feedback_matrix @ vector.T).toarray().ravel(), 0, 1)
        indices = np.flatnonzero(similarities >= 0.80)
        if not len(indices):
            return base

        indices = indices[np.argsort(similarities[indices])[-3:]]
        scores = self.feedback_scores[indices]
        if float(scores.max() - scores.min()) > 3:
            return base

        weights = similarities[indices] ** 4
        feedback = float(np.average(scores, weights=weights))
        similarity = float(similarities[indices].max())
        influence = min(0.35, max(0.0, (similarity - 0.80) / 0.20) * 0.35)
        return float(np.clip(base * (1 - influence) + feedback * influence, 1, 10))

    async def check(self, text: str) -> int:
        async with self._semaphore:
            score = await asyncio.to_thread(self.predict_score, text)
        return 10 if score >= 9.75 else max(1, min(9, math.floor(score)))
