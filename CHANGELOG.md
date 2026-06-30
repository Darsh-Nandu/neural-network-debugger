# Changelog

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
