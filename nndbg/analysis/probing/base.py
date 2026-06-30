"""ProbingAnalyzer — answers "where is concept X encoded?" via linear probes."""
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch

from nndbg.analysis.probing.results import ProbeResult
from nndbg.core.collect import collect_activations

if TYPE_CHECKING:
    from nndbg.inspector import Inspector


class ProbingAnalyzer:
    """Trains a linear probe per layer to test whether a concept is
    linearly decodable from that layer's representation.

    Example::

        dataset = [(tokenizer(text, return_tensors="pt")["input_ids"], label)
                   for text, label in examples]
        result = inspector.probing.fit(dataset, concept="sentiment")
        result.plot()
    """

    def __init__(self, inspector: "Inspector") -> None:
        self._inspector = inspector

    def fit(
        self,
        dataset: Sequence[tuple[torch.Tensor, object]],
        *,
        concept: str,
        layers: list[str] | None = None,
        pooling: str = "last",
        cv: int = 5,
        max_iter: int = 1000,
    ) -> ProbeResult:
        """Train a cross-validated logistic-regression probe at every
        requested layer and report accuracy per layer.

        Args:
            dataset: ``(input_ids, label)`` pairs. Needs at least 2 examples
                per class (``cv`` is shrunk automatically if the smallest
                class has fewer than ``cv`` examples).
            concept: a name for what this probe is testing, used in plots.
            layers: layer names to probe (see ``inspector.layers()``).
                Defaults to every registered layer — for large models,
                passing an explicit, smaller list is faster and more
                interpretable (e.g. one entry per transformer block).
            pooling: how to collapse the sequence dimension before probing
                — ``"last"``, ``"mean"``, or ``"first"`` token.
            cv: number of cross-validation folds.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.preprocessing import LabelEncoder

        inspector = self._inspector
        layer_names = layers or inspector.layers()

        inputs = [item[0] for item in dataset]
        raw_labels = [item[1] for item in dataset]
        encoder = LabelEncoder()
        y = encoder.fit_transform(raw_labels)

        if len(np.unique(y)) < 2:
            raise ValueError("Probing needs at least 2 distinct label values in the dataset.")
        min_class_count = int(np.bincount(y).min())
        if min_class_count < 2:
            raise ValueError("Each class needs at least 2 examples to cross-validate a probe.")
        cv = max(2, min(cv, min_class_count))

        activations = collect_activations(
            inspector.model,
            inspector._hooks,
            inputs,
            layer_names,
            pooling=pooling,
            device=inspector.device,
        )

        accuracies, stds = [], []
        splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=0)
        for name in layer_names:
            X = activations[name].cpu().numpy()
            clf = LogisticRegression(max_iter=max_iter)
            scores = cross_val_score(clf, X, y, cv=splitter)
            accuracies.append(float(scores.mean()))
            stds.append(float(scores.std()))

        return ProbeResult(
            concept=concept,
            layers=layer_names,
            accuracy=accuracies,
            std=stds,
            n_samples=len(dataset),
            n_classes=len(encoder.classes_),
        )
