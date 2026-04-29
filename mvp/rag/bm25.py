"""Lightweight BM25Okapi implementation to avoid external dependency issues.

API-compatible subset of the ``rank_bm25`` package.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


class BM25Okapi:
    """BM25 Okapi ranking function over a pre-tokenised corpus.

    Parameters
    ----------
    corpus : sequence of token lists
        Each element is a list of string tokens for one document.
    k1 : float
        Term-frequency saturation parameter (default 1.5).
    b : float
        Length-normalisation parameter (default 0.75).
    """

    def __init__(
        self,
        corpus: Sequence[Sequence[str]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self._corpus_size = len(corpus)

        self._doc_lens = np.array([len(doc) for doc in corpus], dtype=np.float64)
        self._avgdl = float(self._doc_lens.mean()) if self._corpus_size else 1.0

        self._df: dict[str, int] = {}
        self._tf: list[dict[str, int]] = []

        for doc in corpus:
            tf: dict[str, int] = {}
            seen: set[str] = set()
            for token in doc:
                tf[token] = tf.get(token, 0) + 1
                if token not in seen:
                    self._df[token] = self._df.get(token, 0) + 1
                    seen.add(token)
            self._tf.append(tf)

        self._idf: dict[str, float] = {}
        for word, df in self._df.items():
            self._idf[word] = math.log(
                (self._corpus_size - df + 0.5) / (df + 0.5) + 1.0
            )

    def get_scores(self, query_tokens: Sequence[str]) -> np.ndarray:
        """Return an array of BM25 scores for every document in the corpus."""
        scores = np.zeros(self._corpus_size, dtype=np.float64)
        for q in query_tokens:
            idf = self._idf.get(q, 0.0)
            if idf <= 0:
                continue
            for i, tf_doc in enumerate(self._tf):
                freq = tf_doc.get(q, 0)
                if freq == 0:
                    continue
                dl = self._doc_lens[i]
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                scores[i] += idf * numerator / denominator
        return scores
