# Roadmap

NNDbg v1 ships six analysis planes (probing, attribution, attention,
activation patching, sparse autoencoders, VAE latent analysis), each fully
implemented, tested, and documented. The previous version of this project
tried to scaffold many more planes at once and most never got past empty
stub files — v1 deliberately narrows scope so everything that's exposed
actually works.

The planes below are the deliberate v2 backlog: real, useful additions
that didn't make the v1 cut so the first release could be solid rather
than broad.

## Planned for v2

- **Representation geometry** — CKA (centered kernel alignment) between
  layers or between two models (e.g. base vs. fine-tuned), PCA/UMAP
  projections of representation space over a dataset.
- **Neuron-level analysis** — top-activating examples per neuron,
  dead-neuron detection, a polysemanticity proxy via activation clustering.
- **Concept erasure** — linear concept removal (INLP / LEACE-style
  iterative null-space projection), for testing whether a concept is
  causally necessary once it's no longer linearly decodable.
- **CLI** — `nndbg summary <model>` and friends, for inspecting a model
  without writing a script.
- **Captum integration** — an opt-in `nndbg[captum]` extra wrapping
  Captum's broader attribution method library (DeepLIFT, SHAP, ...) behind
  the same `AttributionResult` interface as the native methods.

## Explicitly out of scope

- A persistent activation-storage backend (the original prototype used
  DuckDB; v1 deliberately keeps activations in memory / plain `torch.save`
  files — add a backend only if a real workflow needs querying activations
  across runs, not preemptively).
