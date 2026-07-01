# Changelog

## 0.2.0 — plane upgrades + three new planes

### Fixes / upgrades to existing planes

- **Probing** — `inspector.probing.fit(..., method=)` now accepts `"logistic"`
  (default, unchanged), `"svm"` (LinearSVC), or `"mlp"` (1-hidden-layer
  MLPClassifier).  `ProbeResult` gains a `method` field shown in `repr()` and
  plot titles.
- **Attribution** — two new methods: `gradient_x_input()` (sign-aware
  saliency) and `smoothgrad()` (averaged saliency over Gaussian-perturbed
  inputs, Smilkov et al. 2017).
- **SAE** — `inspector.sae.train(..., activation=)` now accepts `"topk:<k>"`
  (e.g. `"topk:32"`) for exact top-k sparsity control (Gao et al. 2024),
  in addition to the default `"relu"` L1-penalised mode.
- **Activation patching** — new `inspector.patching.mean_ablation(clean,
  dataset)` method: replaces each (layer, position) with the dataset mean and
  reports the logit drop.  `PatchingResult` gains a `method` field; the plot
  switches to a diverging colormap for mean-ablation results.

### New planes

- **`inspector.geometry`** — `layer_similarity(dataset)` computes pairwise
  linear CKA (Kornblith et al. 2019) between all layers; `compare(other,
  dataset)` cross-compares two model checkpoints on the same data; `pca(dataset,
  layer)` projects a layer's activations to 2-D; `umap(...)` does the same via
  UMAP (requires `pip install umap-learn`).
- **`inspector.neurons`** — `stats(dataset, layer)` reports per-neuron mean,
  max, std, kurtosis (polysemanticity proxy), dead-neuron mask, and top-k
  activating examples.  `top_activating(...)` is a convenience alias.
- **`inspector.erasure`** — `inlp(dataset, concept=, layer=)` runs Iterative
  Null-space Projection (Ravfogel et al. 2020): repeatedly trains a probe and
  projects out its direction, returning an `ErasureResult` with the accumulated
  projection matrix and accuracy decay curve.  `result.apply(activations)`
  projects new activations through the learned erasure.

### Testing

- 30+ new tests covering all new planes and fixed methods (total: ~90 tests).

## 0.1.0 — clean rewrite

The previous `nndbg` codebase had a broken core (`nndbg/core/{hooker,registry,store,tracer}.py`
were empty stub files that the rest of the package imported from) alongside
an orphaned earlier implementation. This release is a full rewrite from
scratch, scoped to six analysis planes that are each genuinely implemented,
tested, and documented, rather than many partial ones.

### Added

- `Inspector` — single entrypoint wrapping any `nn.Module` / HuggingFace
  model, exposing lazily-initialized analysis planes.
- Core hooking infrastructure: `LayerRegistry`, `HookManager`
  (forward-hook capture + activation patching), `ActivationCache`,
  `collect_activations`.
- **Probing** — cross-validated linear probes per layer
  (`inspector.probing.fit`).
- **Attribution** — native saliency, Integrated Gradients, and Grad-CAM
  via autograd (`inspector.attribution.*`), no captum dependency required.
- **Attention analysis** — per-head heatmaps, attention rollout, per-head
  entropy (`inspector.attention.*`).
- **Activation patching / causal tracing** — ROME-style logit-recovery
  heatmaps (`inspector.patching.causal_trace`).
- **Sparse autoencoders** — train + decompose a layer's activations into
  sparse features (`inspector.sae.*`).
- **VAE latent analysis** — compressed latent-space visualization and
  reconstruction-error anomaly detection (`inspector.latent.*`).
- Matplotlib-based plotting (`Result.plot()`) for every plane, with an
  optional interactive Plotly backend (`Result.plotly()`).
- 54 unit tests against a tiny synthetic MLP and a tiny random-init GPT2
  (no network access required).

### Removed

- The DuckDB-backed activation store, the Typer CLI, and the `geometry`/
  `neurons`/`erasure` planes from the previous design — see
  [docs/ROADMAP.md](docs/ROADMAP.md) for the v2 plan.
