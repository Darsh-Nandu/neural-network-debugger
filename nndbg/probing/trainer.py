"""
ProbeTrainer — trains a linear probe per layer per axis.

Core idea:
    For each layer, take the activation stats as features (mean, std,
    l2_norm etc.) and try to predict the group label (english/french/hindi).

    If the probe achieves high accuracy → that layer encodes the concept.
    If the probe fails → that layer doesn't separate the groups.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split

from nndbg.utils import get_logger

logger = get_logger(__name__)

class ProbeTrainer:

    def __init__(self, cv_folds: int = 3, max_iter: int = 1000):
        self.cv_folds = cv_folds
        self.max_iter = max_iter

    def train_on_layer(self, layer_data: Dict[str, List[Dict]]) -> float:
        X, y = self._build_features(layer_data)

        if len(set(y)) < 2:
            return 0.0
        
        # Not enough samples for cross validation
        if len(X) < self.cv_folds * 2:
            return self._simple_accuracy(X, y)
        
        clf = LogisticRegression(
            max_iter=self.max_iter,
            random_state=42,
            C=0.1
        )

        try:
            scores = cross_val_score(
                clf, X, y, cv=self.cv_folds, scoring="accuracy"
            )
            return float(scores.mean())
        except Exception as e:
            logger.warning(f"Probe training failed: {e}")
            return 0.0

    def _build_features(self, layer_data: Dict[str, Dict[str]]) -> Tuple[np.ndarray, np.ndarray]:
        features = []
        labels = []

        for group_name, samples in layer_data.items():
            for sample in samples:
                features.append([
                    sample.get("mean",     0.0),
                    sample.get("std",      0.0),
                    sample.get("l2_norm",  0.0),
                    sample.get("sparsity", 0.0),
                    sample.get("min_val",  0.0),
                    sample.get("max_val",  0.0),
                ])
                labels.append(group_name)
        X = np.array(features, dtype=np.float32)
        y = np.array(labels)

        # Clean any NaN or Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        return X, y
    
    def _simple_accuracy(
            self,
            X: np.ndarray,
            y: np.ndarray,
    ) -> float:
        """Fallback when there are few samples for cross-validation"""

        if len(X) < 4:
            logger.warning(f"Too few samples ({len(X)}) for reliable probe evaluation.")
            return 0.0
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        clf = LogisticRegression(max_iter=self.max_iter, random_state=42)
        try:
            clf.fit(X_train,y_train)
            return float((clf.predict(X_test) == y_test).mean())
        except Exception:
            return 0.0