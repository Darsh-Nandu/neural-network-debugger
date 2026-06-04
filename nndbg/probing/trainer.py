"""
ProbeTrainer — trains a linear probe per layer per axis.

Core idea:
    For each layer, take the activation stats as features (mean, std,
    l2_norm etc.) and try to predict the group label (english/french/hindi).

    If the probe achieves high accuracy → that layer encodes the concept.
    If the probe fails → that layer doesn't separate the groups.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split

from nndbg.utils import get_logger

logger = get_logger(__name__)

# All available features a researcher can choose from
AVAILABLE_FEATURES = ["mean", "std", "l2_norm", "sparsity", "min_val", "max_val"]

class ProbeTrainer:

    def __init__(self, cv_folds: int = 3, max_iter: int = 1000, test_size: float = 0.3, features: Optional[List[str]] = None):
        self.cv_folds = cv_folds
        self.max_iter = max_iter
        self.test_size = test_size

        if features is None:
            self.features = AVAILABLE_FEATURES
        else: 
            invaild = [f for f in features if f not in AVAILABLE_FEATURES]
            if invaild:
                raise ValueError(
                    f"Invalid features: {invaild}."
                    f"Choose from: {AVAILABLE_FEATURES}"
                )
            if len(features) < 1:
                raise ValueError("At least one feature must be specified.")
            self.features = features

        logger.info(
            f"ProbeTrainer ready | "
            f"cv_folds={cv_folds} | "
            f"test_size={test_size} | "
            f"max_iter={max_iter} | "
            f"features={self.features}"
        )

    def train_on_layer(self, layer_data: Dict[str, List[Dict]]) -> float:
        X, y = self._build_features(layer_data)

        if len(set(y)) < 2:
            return 0.0
        
        # Enough samples for cross-validation
        if len(X) >= self.cv_folds * 2:
            clf = LogisticRegression(
                max_iter=self.max_iter,
                random_state=42,
                C=1.0,
            )
            try:
                scores = cross_val_score(
                    clf, X, y,
                    cv=self.cv_folds,
                    scoring="accuracy",
                )
                return float(scores.mean())
            except Exception as e:
                logger.warning(f"Cross-validation failed: {e}")
                return 0.0

        # Fallback: train/test split
        return self._split_accuracy(X, y)

    def train_all_layers(
        self,
        layer_group_data: Dict[str, Dict[str, List[Dict]]],
    ) -> Dict[str, float]:
        """
        Train probes for every layer at once.

        Args:
            layer_group_data:
                {layer_name -> {group_name -> [sample_stats_dict]}}

        Returns:
            {layer_name -> probe_accuracy}
        """
        results = {}
        for layer_name, layer_data in layer_group_data.items():
            score = self.train_on_layer(layer_data)
            results[layer_name] = score
            logger.debug(
                f"Layer '{layer_name}': probe accuracy = {score:.3f}"
            )
        return results

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
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.test_size,
                random_state=42,
                stratify=y,
            )
            clf = LogisticRegression(
                max_iter=self.max_iter,
                random_state=42,
            )
            clf.fit(X_train, y_train)
            return float((clf.predict(X_test) == y_test).mean())
        except Exception as e:
            logger.warning(f"Split accuracy failed: {e}")
            return 0.0
        
    def __repr__(self) -> str:
        return (
            f"ProbeTrainer("
            f"cv_folds={self.cv_folds}, "
            f"test_size={self.test_size}, "
            f"max_iter={self.max_iter}, "
            f"features={self.features})"
        )