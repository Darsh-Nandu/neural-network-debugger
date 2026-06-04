"""
ProbeTrainer — trains a configurable probe per layer per axis.

Supported probe types
---------------------
  "logistic"      LogisticRegression  (default) — fast, interpretable linear probe
  "ridge"         RidgeClassifier     — regularised linear, good for high-dim features
  "svm"           LinearSVC           — maximum-margin linear probe
  "mlp"           MLPClassifier       — two-layer non-linear probe
  "knn"           KNeighborsClassifier — distance-based, no training phase
  "random_forest" RandomForestClassifier — ensemble of decision trees

"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split

from nndbg.utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AVAILABLE_FEATURES = ["mean", "std", "l2_norm", "sparsity", "min_val", "max_val"]

PROBE_TYPES = [
    "logistic",       # LogisticRegression
    "ridge",          # RidgeClassifier
    "svm",            # LinearSVC
    "mlp",            # MLPClassifier
    "knn",            # KNeighborsClassifier
    "random_forest",  # RandomForestClassifier
]


# ---------------------------------------------------------------------------
# ProbeTrainer
# ---------------------------------------------------------------------------

class ProbeTrainer:
    """
    Train classification probes per layer to measure concept encoding.

    Args:
        probe_type:   Which classifier to use. One of PROBE_TYPES.
                      Default: "logistic".
        cv_folds:     Number of cross-validation folds. Default: 3.
        max_iter:     Max solver iterations (used by logistic / ridge / svm / mlp).
                      Default: 1000.
        test_size:    Fraction used for the train/test-split fallback when
                      there are too few samples for cross-validation. Default: 0.3.
        features:     Subset of activation statistics to use as input features.
                      Must be a list drawn from AVAILABLE_FEATURES.
                      Default: all six features.

    Examples:
        # Default logistic probe
        trainer = ProbeTrainer()

        # SVM probe, custom features
        trainer = ProbeTrainer(probe_type="svm", features=["mean", "std", "l2_norm"])

        # Random forest, more folds
        trainer = ProbeTrainer(probe_type="random_forest", cv_folds=5)

        # MLP with extended training budget
        trainer = ProbeTrainer(probe_type="mlp", max_iter=5000)

        # KNN — max_iter / cv_folds still accepted, just unused by KNN itself
        trainer = ProbeTrainer(probe_type="knn")
    """

    def __init__(
        self,
        probe_type: str = "logistic",
        cv_folds: int = 3,
        max_iter: int = 1000,
        test_size: float = 0.3,
        features: Optional[List[str]] = None,
    ):
        if probe_type not in PROBE_TYPES:
            raise ValueError(
                f"Unknown probe_type '{probe_type}'. "
                f"Choose from: {PROBE_TYPES}"
            )

        self.probe_type = probe_type
        self.cv_folds = cv_folds
        self.max_iter = max_iter
        self.test_size = test_size

        if features is None:
            self.features = AVAILABLE_FEATURES[:]
        else:
            invalid = [f for f in features if f not in AVAILABLE_FEATURES]
            if invalid:
                raise ValueError(
                    f"Invalid features: {invalid}. "
                    f"Choose from: {AVAILABLE_FEATURES}"
                )
            if len(features) < 1:
                raise ValueError("At least one feature must be specified.")
            self.features = features

        logger.info(
            f"ProbeTrainer ready | "
            f"probe_type={probe_type} | "
            f"cv_folds={cv_folds} | "
            f"test_size={test_size} | "
            f"max_iter={max_iter} | "
            f"features={self.features}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_on_layer(self, layer_data: Dict[str, List[Dict]]) -> float:
        """
        Train a probe on a single layer's activation stats.

        Args:
            layer_data: {group_name -> [sample_stats_dict, ...]}

        Returns:
            probe accuracy in [0, 1]
        """
        X, y = self._build_features(layer_data)

        if len(set(y)) < 2:
            return 0.0

        clf = self._make_classifier()

        # Enough samples for cross-validation?
        if len(X) >= self.cv_folds * 2:
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

        # Fallback: simple train/test split
        return self._split_accuracy(X, y)

    def train_all_layers(
        self,
        layer_group_data: Dict[str, Dict[str, List[Dict]]],
    ) -> Dict[str, float]:
        """
        Train probes for every layer at once.

        Args:
            layer_group_data:
                {layer_name -> {group_name -> [sample_stats_dict, ...]}}

        Returns:
            {layer_name -> probe_accuracy}
        """
        results = {}
        for layer_name, layer_data in layer_group_data.items():
            score = self.train_on_layer(layer_data)
            results[layer_name] = score
            logger.debug(f"Layer '{layer_name}': probe accuracy = {score:.3f}")
        return results

    # ------------------------------------------------------------------
    # Classifier factory
    # ------------------------------------------------------------------

    def _make_classifier(self):
        """Instantiate the chosen classifier with current hyperparameters."""
        if self.probe_type == "logistic":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(
                max_iter=self.max_iter,
                random_state=42,
                C=1.0,
            )

        elif self.probe_type == "ridge":
            from sklearn.linear_model import RidgeClassifier
            # RidgeClassifier uses alpha (not C), max_iter optional
            return RidgeClassifier(
                max_iter=self.max_iter,
                alpha=1.0,
            )

        elif self.probe_type == "svm":
            from sklearn.svm import LinearSVC
            return LinearSVC(
                max_iter=self.max_iter,
                random_state=42,
                C=1.0,
                dual="auto",
            )

        elif self.probe_type == "mlp":
            from sklearn.neural_network import MLPClassifier
            # early_stopping requires numeric labels; disabled here because
            # group names are arbitrary strings. max_iter caps training instead.
            return MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=self.max_iter,
                random_state=42,
                early_stopping=False,
            )

        elif self.probe_type == "knn":
            from sklearn.neighbors import KNeighborsClassifier
            # n_neighbors capped at sample count later via cross_val_score;
            # max_iter / cv_folds still accepted at construction, unused by KNN itself.
            return KNeighborsClassifier(n_neighbors=5, metric="euclidean")

        elif self.probe_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
            )

        else:  # should never happen — validated in __init__
            raise ValueError(f"Unknown probe_type: {self.probe_type}")

    # ------------------------------------------------------------------
    # Feature building
    # ------------------------------------------------------------------

    def _build_features(
        self,
        layer_data: Dict[str, List[Dict]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert {group -> [sample_stats]} into (X, y) arrays.
        Only the features listed in self.features are included.
        """
        features_list = []
        labels = []

        for group_name, samples in layer_data.items():
            for sample in samples:
                row = [sample.get(f, 0.0) for f in self.features]
                features_list.append(row)
                labels.append(group_name)

        X = np.array(features_list, dtype=np.float32)
        y = np.array(labels)

        # Sanitise NaN / Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        return X, y

    # ------------------------------------------------------------------
    # Fallback evaluation
    # ------------------------------------------------------------------

    def _split_accuracy(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """Train/test split fallback when samples are too few for CV."""
        if len(X) < 4:
            logger.warning(
                f"Too few samples ({len(X)}) for reliable probe evaluation."
            )
            return 0.0

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.test_size,
                random_state=42,
                stratify=y,
            )
            clf = self._make_classifier()
            clf.fit(X_train, y_train)
            return float((clf.predict(X_test) == y_test).mean())
        except Exception as e:
            logger.warning(f"Split accuracy failed: {e}")
            return 0.0

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ProbeTrainer("
            f"probe_type='{self.probe_type}', "
            f"cv_folds={self.cv_folds}, "
            f"test_size={self.test_size}, "
            f"max_iter={self.max_iter}, "
            f"features={self.features})"
        )