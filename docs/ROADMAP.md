# Roadmap

NNDbg ships fully-implemented, tested analysis planes rather than broad
scaffolding. Each version ships what it ships completely.

## Shipped in v0.2.0 (new planes)

- **Representation geometry** (`inspector.geometry`) — linear CKA between
  layers, cross-model comparison, PCA and optional UMAP projections.
- **Neuron-level analysis** (`inspector.neurons`) — dead-neuron detection,
  top-activating examples, per-neuron kurtosis as a polysemanticity proxy.
- **Concept erasure** (`inspector.erasure`) — INLP iterative null-space
  projection; returns a projection matrix applicable to new activations.

Also in v0.2.0 (upgrades):
- Probing: `method="svm"` / `"mlp"` options.
- Attribution: `gradient_x_input()` and `smoothgrad()`.
- SAE: `activation="topk:<k>"` for exact sparsity.
- Patching: `mean_ablation(clean, dataset)`.

## Planned for v3

- **CLI** — `nndbg summary <model>` and friends, for inspecting a model
  without writing a script.
- **Captum integration** — an opt-in `nndbg[captum]` extra wrapping
  Captum's broader attribution method library (DeepLIFT, SHAP, ...) behind
  the same `AttributionResult` interface as the native methods.
- **LEACE concept erasure** — closed-form concept removal with better
  theoretical guarantees than INLP for multiclass concepts.

## Explicitly out of scope

- A persistent activation-storage backend (the original prototype used
  DuckDB; activations live in memory / plain `torch.save` files — add a
  backend only if a real workflow needs querying activations across runs,
  not preemptively).
