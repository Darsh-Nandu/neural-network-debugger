# NNDbg — Neural Network Semantic Activation Analyzer

<p align="center">
  <img alt="Version" src="https://img.shields.io/badge/version-0.1.0-blue?style=flat-square">
  <img alt="Python" src="https://img.shields.io/badge/python-%3E%3D3.10-blue?style=flat-square&logo=python">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-%3E%3D2.0-orange?style=flat-square&logo=pytorch">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green?style=flat-square">
  <img alt="Status" src="https://img.shields.io/badge/status-alpha-yellow?style=flat-square">
</p>

> **Deep semantic activation analysis for neural networks.** Feed any PyTorch or HuggingFace model contrasting inputs — languages, domains, sentiments — and discover *exactly which layers* encode those concepts internally.

---

## What Is NNDbg?

NNDbg is a neural network interpretability library built for researchers and practitioners who want to **look inside** a model without modifying its code.

It answers questions like:

- **Which transformer layer first learns to distinguish English from French?**
- **Where does medical knowledge separate from legal knowledge?**
- **How does fine-tuning change where sentiment is encoded?**
- **At what layer does a sentence's internal representation stabilize?**

NNDbg does this by dynamically attaching forward hooks to every named layer of any PyTorch model, capturing activation statistics, persisting them to a DuckDB database, training **probes** (classifiers) per layer, and using probe accuracy as a measure of how well each layer encodes a given concept.

---

## What's New in This Release

| Change | Summary |
|---|---|
| **Silent by default** | No terminal output unless you call `nndbg.set_verbose(True)` |
| **6 probe types** | Choose from `logistic`, `ridge`, `svm`, `mlp`, `knn`, `random_forest` via the `probe_type` parameter |
| **`results.save(path)`** | Export everything — probe scores, activation stats, and raw tensors — to a zip archive |
| **Test script** | `test.py` ships with config flags at the top; no code editing required |
| **Bug fixes** | `_build_features` now respects the `features` argument; `_split_accuracy` naming fixed |

---

## Key Features

| Feature | Description |
|---|---|
| **Zero-code model instrumentation** | Attach hooks to any PyTorch model without touching its source |
| **Axis API** | Define semantic dimensions (language, domain, sentiment, etc.) with labeled text groups |
| **6 probe types** | Linear, ridge, SVM, MLP, KNN, and random forest probes — all swappable with one parameter |
| **Silent by default** | Fully quiet during `run()` unless verbose mode is enabled |
| **Zip export** | Save all results including raw activation tensors to a portable zip file |
| **Activation trajectory** | Trace a single input through all layers and watch how its representation evolves |
| **Sentence comparison** | Overlay two inputs' activation paths to spot where they diverge |
| **DuckDB persistence** | Store runs to disk for later querying; default is in-memory |
| **Interactive Plotly dashboards** | Bar charts, heatmaps, and trajectory plots with hover details |

---

## Architecture Overview

```
nndbg/
├── probe.py              # ModelProbe — main entry point
├── results.py            # ProbeResults — output, visualizations, save()
│
├── hooks/
│   ├── engine.py         # HookEngine — dynamic forward hook attachment & capture
│   └── registry.py       # HookRegistry — layer metadata
│
├── storage/
│   └── store.py          # ActivationStore — DuckDB-backed activation persistence
│
├── probing/
│   ├── axis.py           # Axis — semantic comparison dimension dataclass
│   └── trainer.py        # ProbeTrainer — 6 classifier types, trains per layer
│
├── attribution/          # (Phase 2) Advanced attribution tools
│   ├── neuron.py
│   ├── head.py
│   ├── interference.py
│   └── drift.py
│
├── visualization/
│   └── trajectory.py     # TraceResults + LayerTrace — single-input trajectory
│
└── utils/
    ├── logging.py        # Structured logger with set_verbose() control
    ├── device.py         # Auto device selection (CPU/CUDA/MPS)
    └── console.py        # Rich console instance
```

### How It Works: Step by Step

```
ModelProbe.run()
│
├── 1. HookEngine.attach()
│      Registers a forward hook on every named nn.Module in the model.
│
├── 2. Forward pass per sample (silent by default; set_verbose(True) to see progress)
│      For each (axis, group, text):
│        - Tokenize → run model(**inputs) under the capture context
│        - Collect activation stats (mean, std, l2_norm, sparsity, min, max)
│        - Store raw activation tensors (available via results.save())
│        - Persist stats to DuckDB
│
├── 3. HookEngine.detach()
│
├── 4. ProbeTrainer.train_all_layers()
│      For each layer × axis:
│        - Build feature matrix X from activation stats
│        - Train the chosen classifier (logistic / ridge / svm / mlp / knn / rf)
│        - Record accuracy as the "probe score"
│
└── 5. Return ProbeResults
       Contains probe_scores, layer_group_data, raw_activations, and all
       visualization + export methods.
```

---

## Installation

### Prerequisites

- Python ≥ 3.10
- PyTorch ≥ 2.0

### From Source

```bash
git clone https://github.com/Darsh-Nandu/nndbg.git
cd nndbg

# With Poetry (recommended)
poetry install

# Or with pip
pip install -e .
```

### Dependencies

| Package | Purpose |
|---|---|
| `torch ≥ 2.0` | Model forward passes and tensor operations |
| `transformers ≥ 4.35` | HuggingFace model + tokenizer loading |
| `duckdb ≥ 0.9` | Activation storage backend |
| `scikit-learn ≥ 1.3` | All probe classifiers |
| `plotly ≥ 5.17` | Interactive visualizations |
| `rich ≥ 13.0` | Terminal tables and reports |
| `numpy ≥ 1.24` | Numerical operations |
| `tqdm ≥ 4.65` | Progress bars (shown only in verbose mode) |

---

## Quick Start

### Running `test.py`

The bundled `test.py` is a ready-to-run script with a config block at the top:

```python
VERBOSE     = True          # True → logs + progress bars / False → silent
PROBE_TYPE  = "logistic"    # logistic | ridge | svm | mlp | knn | random_forest
SAVE_PATH   = "results.zip" # None to skip saving
```

```bash
python test.py
```

### Verbose Mode

NNDbg is **completely silent by default**. Enable logging and progress bars with one call before building your probe:

```python
import nndbg

nndbg.set_verbose(True)    # show INFO logs + tqdm progress bars
nndbg.set_verbose(False)   # (default) silent
print(nndbg.is_verbose())  # check current state
```

### Basic Example — Language Axis

```python
import nndbg
from nndbg import ModelProbe

nndbg.set_verbose(True)   # optional: see progress during run()

probe = ModelProbe.from_pretrained("bert-base-multilingual-cased")

probe.add_axis("language", {
    "english": [
        "The stock market crashed today.",
        "Scientists discovered a new species.",
        "The election results were announced.",
    ],
    "french": [
        "Le marché boursier s'est effondré aujourd'hui.",
        "Les scientifiques ont découvert une nouvelle espèce.",
        "Les résultats des élections ont été annoncés.",
    ],
    "hindi": [
        "आज शेयर बाजार गिर गया।",
        "वैज्ञानिकों ने एक नई प्रजाति की खोज की।",
        "चुनाव परिणाम घोषित किए गए।",
    ],
})

results = probe.run()

# Print a rich terminal report
results.summary()

# Interactive Plotly dashboard
results.show()

# Save everything (scores + raw tensors) to a zip
results.save("language_run.zip")
```

### Choosing a Probe Type

Pass `probe_type` to `ProbeTrainer`. All existing arguments still work unchanged:

```python
from nndbg import ModelProbe, ProbeTrainer

trainer = ProbeTrainer(
    probe_type = "svm",           # logistic | ridge | svm | mlp | knn | random_forest
    cv_folds   = 5,
    max_iter   = 2000,
    test_size  = 0.2,
    features   = ["mean", "std", "l2_norm"],
)

probe = ModelProbe.from_pretrained(
    "bert-base-multilingual-cased",
    probe_trainer=trainer,
)
```

### Saving Results to a Zip

```python
results = probe.run()

# Saves probe scores, activation stats, and raw .npy tensors
path = results.save("my_experiment.zip")
print(f"Saved → {path}")
```

The zip archive contains:

```
my_experiment.zip
├── metadata.json              run ID, model name, axes
├── probe_scores.json          {axis → {layer → accuracy}}
├── activation_stats.json      per-layer per-group aggregated stats
├── layer_group_data.json      per-sample stats (mean/std/…)
└── activations/
    └── <axis>/<group>/sample_NNNN/<layer>.npy   ← raw float32 tensors
```

### Multi-Axis Example

```python
probe = ModelProbe.from_pretrained("google/mt5-small")

probe \
    .add_axis("language", {
        "english": ["..."] * 10,
        "french":  ["..."] * 10,
    }) \
    .add_axis("domain", {
        "legal":   ["The defendant shall appear before the court.", ...],
        "medical": ["Patient presents with acute symptoms.", ...],
        "code":    ["def forward(self, x): return self.layer(x)", ...],
    }) \
    .add_axis("sentiment", {
        "positive": ["Great product! Absolutely loved it.", ...],
        "negative": ["Terrible experience. Worst purchase ever.", ...],
    })

results = probe.run()
results.summary(top_k=5)
results.save("multi_axis_run.zip")
```

### Activation Trajectory (Single Input)

```python
probe = ModelProbe.from_pretrained("bert-base-multilingual-cased")

trace = probe.trace("Le chat était assis sur le tapis.")
trace.summary()
trace.show()

print(trace.most_active(top_k=5))
print(trace.stable_at())

tensor = trace.activations["encoder.layer.4"]
print(tensor.shape)
```

### Comparing Two Sentences

```python
english = probe.trace("The cat sat on the mat.")
french  = probe.trace("Le chat était assis sur le tapis.")

# Overlays both trajectories; marks layer of maximum divergence
english.compare(french)
```

### Using an Existing Model

```python
import torch.nn as nn
from nndbg import ModelProbe

probe = ModelProbe.from_model(
    my_model,
    tokenizer=my_tokenizer,
    model_name="my-custom-model",
)
```

---

## Understanding the Output

### Probe Score Matrix

After `probe.run()`, the key output is a matrix of **probe scores** — one per layer per axis:

```
               language  domain  sentiment
encoder.0        0.51     0.48     0.52     ← near chance, encodes nothing yet
encoder.4        0.89     0.61     0.58     ← language crystallizes at layer 4
encoder.8        0.87     0.93     0.64     ← domain separates strongly at layer 8
encoder.12       0.82     0.88     0.91     ← sentiment appears late
```

| Score | Interpretation |
|---|---|
| ≥ 0.8 | Layer **strongly** encodes the concept |
| 0.6–0.8 | Moderate encoding |
| ≤ 0.6 | Near chance — layer doesn't meaningfully encode the concept |

### Activation Features

For each layer and sample, NNDbg extracts 6 statistics from the raw activation tensor:

| Feature | Meaning |
|---|---|
| `mean` | Average activation value |
| `std` | Standard deviation |
| `l2_norm` | Euclidean norm — total magnitude |
| `sparsity` | Fraction of neurons outputting exactly 0 |
| `min_val` | Minimum activation |
| `max_val` | Maximum activation |

### Available Probe Types

| `probe_type` | Classifier | Best for |
|---|---|---|
| `"logistic"` | `LogisticRegression` | Default — fast, interpretable |
| `"ridge"` | `RidgeClassifier` | High-dimensional or collinear features |
| `"svm"` | `LinearSVC` | Large sample counts, max-margin boundary |
| `"mlp"` | `MLPClassifier (64→32)` | Non-linear concept boundaries |
| `"knn"` | `KNeighborsClassifier` | Distribution-agnostic; no training phase |
| `"random_forest"` | `RandomForestClassifier` | Interaction effects between features |

---

## API Reference

### `nndbg` (top-level)

```python
import nndbg

nndbg.set_verbose(True)     # enable logs + progress bars
nndbg.set_verbose(False)    # (default) silent
nndbg.is_verbose()          # → bool
```

### `ModelProbe`

```python
# From HuggingFace
probe = ModelProbe.from_pretrained(model_name, store_path=":memory:", max_length=128)

# From an existing PyTorch model
probe = ModelProbe.from_model(model, tokenizer, model_name="custom")

probe.add_axis(name, groups_dict)   # chainable, returns self
results = probe.run()               # → ProbeResults
trace   = probe.trace(text)         # → TraceResults
arch    = probe.architecture()      # → Dict
```

### `ProbeResults`

```python
results.summary(top_k=5, min_score=0.0)         # rich terminal report
results.show(top_k=None, min_score=0.0)          # interactive Plotly bar charts
results.plot_heatmap(min_score=0.0)              # layer × axis heatmap
results.encoding_layers(axis, top_k=5, min_score=0.0)  # → List[Tuple[str, float]]
results.save(path)                               # → str (zip archive path)
results.to_dict()                                # → dict
```

### `TraceResults`

```python
trace.summary()                   # rich terminal layer table
trace.show()                      # interactive Plotly trajectory chart
trace.most_active(top_k=5)        # → List[Tuple[str, float]]
trace.stable_at(threshold=0.01)   # → Optional[str]
trace.compare(other_trace)        # overlay chart with divergence marker
trace.activations                 # Dict[str, torch.Tensor]
```

### `ProbeTrainer`

```python
from nndbg import ProbeTrainer

trainer = ProbeTrainer(
    probe_type = "logistic",     # see table above; default: "logistic"
    cv_folds   = 3,              # cross-validation folds
    max_iter   = 1000,           # solver iterations (logistic/ridge/svm/mlp)
    test_size  = 0.3,            # fallback split when too few samples
    features   = None,           # None = all 6; or a subset list
)
```

Available features: `mean`, `std`, `l2_norm`, `sparsity`, `min_val`, `max_val`

---

## Running Tests

```bash
# pytest suite (95 tests, no HuggingFace download required)
PYTHONPATH=. pytest tests/test_nndbg.py -v --override-ini="addopts="

# Runnable integration script (downloads bert-base-multilingual-cased)
python test.py
```

---

## Roadmap

### Phase 1 — Core Activation Analysis *(current)*
- [x] `HookEngine` — dynamic hook attachment/detach
- [x] `ActivationStore` — DuckDB-backed storage
- [x] `Axis` API — semantic comparison dimensions
- [x] `ProbeTrainer` — 6 probe types, all existing args preserved
- [x] Silent by default with `set_verbose()` toggle
- [x] `results.save()` — zip export with raw tensors
- [x] `ProbeResults` — terminal report + Plotly dashboards
- [x] `TraceResults` — single-input activation trajectory
- [x] Sentence comparison with divergence detection

### Phase 2 — Neuron & Head Attribution
- [x] `NeuronAttributor` — top-k neurons per concept group
- [x] `HeadAttributor` — attention head specialization map
- [ ] Dead neuron detection

### Phase 3 — Advanced Analysis
- [x] Concept interference detection
- [ ] Activation trajectory drift (pre/post fine-tuning)
- [ ] Concept representation evolution across training steps

### Phase 4 — Diagnostics & Reporting
- [ ] Gradient monitoring
- [ ] HTML/PDF report export
- [ ] CLI interface (`nndbg` command)

---

## Development

```bash
# Run tests
PYTHONPATH=. pytest tests/test_nndbg.py -v --override-ini="addopts="

# Format
poetry run black nndbg/
poetry run ruff check nndbg/
```

---

## License

MIT — see [LICENSE](LICENSE) for details.