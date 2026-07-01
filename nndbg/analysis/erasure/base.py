"""ErasureAnalyzer — causally remove a concept from a layer's representations.

Implements Iterative Null-space Projection (INLP, Ravfogel et al. 2020):
train a linear probe, project out its weight direction from the activations,
repeat until the probe can no longer decode the concept above chance.  The
accumulated projection matrix can then be applied to new activations to
erase the concept for downstream analysis.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch

from nndbg.analysis.erasure.results import ErasureResult
from nndbg.core.collect import collect_activations

if TYPE_CHECKING:
    from nndbg.inspector import Inspector


class ErasureAnalyzer:
    """Concept erasure via Iterative Null-space Projection (INLP).

    Example::

        result = inspector.erasure.inlp(dataset, concept="sentiment", layer="transformer.h.4")
        result.plot()              # probe accuracy decays toward chance
        print(result)              # 8 iters, 0.95 -> 0.51

        # erase the concept from new activations
        erased = result.apply(some_activation_tensor)
    """

    def __init__(self, inspector: Inspector) -> None:
        self._inspector = inspector

    def inlp(
        self,
        dataset: Sequence[tuple[torch.Tensor, object]],
        *,
        concept: str,
        layer: str,
        pooling: str = "last",
        n_iters: int = 20,
        min_accuracy: float = 0.55,
        max_iter: int = 1000,
    ) -> ErasureResult:
        """Run INLP to erase the linearly-decodable signal for ``concept``.

        Each iteration:
        1. Train a logistic-regression probe on the current (projected) X.
        2. If train accuracy < ``min_accuracy``, stop — the concept is gone.
        3. Extract the probe's weight direction(s).
        4. Project those directions out of X (null-space projection).
        5. Accumulate the projection into the master matrix P.

        Args:
            dataset: ``(input_ids, label)`` pairs (same format as probing).
            concept: name of the concept being erased, used in plots.
            layer: layer whose activations to erase the concept from.
            n_iters: maximum number of projection iterations.
            min_accuracy: stop when probe train accuracy drops below this
                (0.55 = slightly above chance for binary, adjust for more
                classes).
            max_iter: max iterations for each inner LogisticRegression fit.

        Returns:
            ErasureResult with the accumulated projection matrix and the
            accuracy decay curve.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import LabelEncoder

        inspector = self._inspector
        inputs = [item[0] for item in dataset]
        raw_labels = [item[1] for item in dataset]
        encoder = LabelEncoder()
        y = encoder.fit_transform(raw_labels)

        acts = collect_activations(
            inspector.model, inspector._hooks, inputs, [layer], pooling=pooling, device=inspector.device
        )
        X = acts[layer].float().cpu().numpy()
        d = X.shape[1]
        P = np.eye(d, dtype=np.float32)

        accuracy_trace: list[float] = []
        actual_iters = 0

        for _ in range(n_iters):
            clf = LogisticRegression(max_iter=max_iter, solver="lbfgs", multi_class="auto")
            clf.fit(X, y)
            acc = float(clf.score(X, y))
            accuracy_trace.append(acc)
            if acc < min_accuracy:
                break

            # For binary (coef_ shape (1,d)) or multiclass ((n_classes,d)),
            # use SVD of the coefficient matrix to get the most-predictive
            # directions; remove one direction per iteration.
            W = clf.coef_  # (n_classes_or_1, d)
            if W.shape[0] == 1:
                w = W[0]
            else:
                # first right singular vector spans the most-predictive direction
                _, _, Vt = np.linalg.svd(W, full_matrices=False)
                w = Vt[0]

            w = w / (np.linalg.norm(w) + 1e-10)
            Pw = np.eye(d, dtype=np.float32) - np.outer(w, w)
            P = Pw @ P
            X = X @ Pw.T
            actual_iters += 1

        return ErasureResult(
            concept=concept,
            layer=layer,
            projection=torch.from_numpy(P),
            accuracy_trace=accuracy_trace,
            n_iters=actual_iters,
        )
