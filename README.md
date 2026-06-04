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
- **Where does medical knowledge separate from legal knowledge in a model?**
- **How does fine-tuning change where sentiment is encoded?**
- **Which neurons fire most strongly for a specific concept group?**
- **At what layer does a sentence's internal representation stabilize?**

NNDbg does this by dynamically attaching forward hooks to every named layer of any PyTorch model, capturing activation statistics, persisting them to a DuckDB database, and training **linear probes** (logistic regression classifiers) per layer — using probe accuracy as a measure of how well each layer encodes a given concept.

---

## Key Features

| Feature | Description |
|---|---|
| **Zero-code model instrumentation** | Attach hooks to any PyTorch model without touching its source |
| **Axis API** | Define semantic dimensions (language, domain, sentiment, etc.) with labeled text groups |
| **Linear probing** | Per-layer concept encoding measured via logistic regression accuracy |
| **Activation trajectory** | Trace a single input through all layers and watch how its representation evolves |
| **Sentence comparison** | Overlay two inputs' activation paths to spot where they diverge |
| **DuckDB persistence** | Store runs to disk for later querying; default is in-memory |
| **Rich terminal output** | Beautiful, color-coded tables and reports in the terminal |
| **Interactive Plotly dashboards** | Bar charts, heatmaps, and trajectory plots with hover details |
| **Neuron & Head attribution** | (Phase 2) Pinpoint which neurons and attention heads carry a concept |
| **Interference detection** | (Phase 2) Find neurons shared across concept axes |

---

## Architecture Overview

```
nndbg/
├── probe.py              # ModelProbe — the main entry point
├── results.py            # ProbeResults — analysis output + visualizations
│
├── hooks/
│   ├── engine.py         # HookEngine — dynamic forward hook attachment & capture
│   └── registry.py       # HookRegistry — layer metadata (type, depth, param count)
│
├── storage/
│   └── store.py          # ActivationStore — DuckDB-backed activation persistence
│
├── probing/
│   ├── axis.py           # Axis — dataclass for semantic comparison dimensions
│   └── trainer.py        # ProbeTrainer — trains logistic regression per layer
│
├── attribution/          # (Phase 2) Advanced attribution tools
│   ├── neuron.py         # NeuronAttributor — top-k neurons per concept
│   ├── head.py           # HeadAttributor — attention head specialization
│   ├── interference.py   # InterferenceDetector — shared neuron detection
│   └── drift.py          # Fine-tune drift analysis
│
├── visualization/
│   └── trajectory.py     # TraceResults + LayerTrace — single-input trajectory
│
└── utils/
    ├── logging.py        # Structured logger
    ├── device.py         # Auto device selection (CPU/CUDA/MPS)
    └── console.py        # Rich console instance
```

### How It Works: Step by Step

```
ModelProbe.run()
│
├── 1. HookEngine.attach()
│      Registers a forward hook on EVERY named nn.Module in the model.
│      Each hook captures the output tensor and stores it under the layer name.
│
├── 2. Forward pass per sample
│      For each (axis, group, text) triple:
│        - Tokenize with the model's tokenizer
│        - Run model(**inputs) under the capture context
│        - Collect activation stats (mean, std, l2_norm, sparsity, min, max)
│        - Persist stats to DuckDB (ActivationStore)
│
├── 3. HookEngine.detach()
│      Removes all hooks cleanly.
│
├── 4. ProbeTrainer.train_all_layers()
│      For each layer × each axis:
│        - Build feature matrix X from activation stats
│        - Build label vector y from group names
│        - Train LogisticRegression with k-fold cross-validation
│        - Record accuracy as the "probe score"
│
└── 5. Return ProbeResults
       Contains the full probe_scores matrix and all visualization methods.
```

---

## Installation

### Prerequisites

- Python ≥ 3.10
- PyTorch ≥ 2.0

### From Source (recommended for now)

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
| `scikit-learn ≥ 1.3` | Logistic regression probe training |
| `plotly ≥ 5.17` | Interactive visualizations |
| `rich ≥ 13.0` | Terminal tables and reports |
| `numpy ≥ 1.24` | Numerical operations |
| `tqdm ≥ 4.65` | Progress bars |

---

## Quick Start

### Basic Example — Language Axis

```python
from nndbg import ModelProbe

# Load any HuggingFace model
probe = ModelProbe.from_pretrained("bert-base-multilingual-cased")

# Define a comparison axis: which groups differ, and how?
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

# Run the full analysis
results = probe.run()

# Print a rich terminal report
results.summary()

# Open an interactive Plotly dashboard
results.show()

# Layer × axis encoding heatmap
results.plot_heatmap()
```

### Multi-Axis Example

```python
probe = ModelProbe.from_pretrained("google/mt5-small")

# Chain multiple axes
probe.add_axis("language", {
    "english": ["..."] * 10,
    "french":  ["..."] * 10,
}).add_axis("domain", {
    "legal":   ["The defendant shall appear before the court.", ...],
    "medical": ["Patient presents with acute symptoms.", ...],
    "code":    ["def forward(self, x): return self.layer(x)", ...],
}).add_axis("sentiment", {
    "positive": ["Great product! Absolutely loved it.", ...],
    "negative": ["Terrible experience. Worst purchase ever.", ...],
    "neutral":  ["It arrived on time. Standard packaging.", ...],
})

results = probe.run()
results.summary(top_k=5)
```

### Activation Trajectory (Single Input)

```python
probe = ModelProbe.from_pretrained("bert-base-multilingual-cased")

# Trace a single sentence through all layers
trace = probe.trace("Le chat était assis sur le tapis.")

# Terminal report: all layers with stats
trace.summary()

# Interactive Plotly line chart
trace.show()

# Top-5 most active layers by L2 norm
print(trace.most_active(top_k=5))

# Find where the representation stabilizes
print(trace.stable_at())

# Access raw tensors
tensor = trace.activations["encoder.layer.4"]
print(tensor.shape)
```

### Comparing Two Sentences

```python
english = probe.trace("The cat sat on the mat.")
french  = probe.trace("Le chat était assis sur le tapis.")

# Overlay both activation trajectories on the same chart
# Marks the layer of maximum divergence
english.compare(french)
```

### Using an Existing Model

```python
import torch.nn as nn
from nndbg import ModelProbe

my_model = ...      # any nn.Module
my_tokenizer = ...  # any HuggingFace tokenizer

probe = ModelProbe.from_model(
    my_model,
    tokenizer=my_tokenizer,
    model_name="my-custom-model",
)
```

### Custom ProbeTrainer

```python
from nndbg.probing.trainer import ProbeTrainer

trainer = ProbeTrainer(
    cv_folds=5,           # number of cross-validation folds
    max_iter=2000,        # logistic regression max iterations
    test_size=0.2,        # fallback train/test split ratio
    features=["mean", "std"],  # subset of activation stats to use as features
)

probe = ModelProbe.from_pretrained(
    "bert-base-multilingual-cased",
    probe_trainer=trainer,
)
```

### Persisting Runs to Disk

```python
probe = ModelProbe.from_pretrained(
    "google/mt5-small",
    store_path="./my_run.duckdb",   # persists to disk instead of memory
    max_length=256,
    batch_size=16,
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

- **Score ≥ 0.8** — Layer strongly encodes the concept
- **Score 0.6–0.8** — Moderate encoding
- **Score ≤ 0.6** — Near chance; layer does not meaningfully encode the concept

### Activation Features Used by the Probe

For each layer and sample, NNDbg extracts these statistics from the raw activation tensor:

| Feature | Meaning |
|---|---|
| `mean` | Average activation value across all neurons |
| `std` | Standard deviation — how spread out activations are |
| `l2_norm` | Euclidean norm — total activation magnitude |
| `sparsity` | Fraction of neurons outputting exactly 0 |
| `min_val` | Minimum activation value |
| `max_val` | Maximum activation value |

The logistic regression probe uses these 6 numbers as a feature vector to try to predict the group label. High classification accuracy = the layer's internal geometry separates the groups.

---

## API Reference

### `ModelProbe`

```python
# From HuggingFace
probe = ModelProbe.from_pretrained(model_name, store_path=":memory:", max_length=128)

# From an existing PyTorch model
probe = ModelProbe.from_model(model, tokenizer, model_name="custom")

# Add an axis (chainable)
probe.add_axis(name, groups_dict)

# Run full analysis → ProbeResults
results = probe.run()

# Trace a single input → TraceResults
trace = probe.trace(text)

# Get architecture summary
arch = probe.architecture()
```

### `ProbeResults`

```python
# Rich terminal report
results.summary(top_k=5, min_score=0.0)

# Interactive Plotly bar charts (one per axis)
results.show(top_k=None, min_score=0.0)

# Layer × axis heatmap
results.plot_heatmap(min_score=0.0)

# Top-k layers for a given axis
results.encoding_layers("language", top_k=5, min_score=0.7)

# Export as plain dict
results.to_dict()
```

### `TraceResults`

```python
# Rich terminal summary of all layer stats
trace.summary()

# Interactive Plotly trajectory (mean, std, l2 per layer)
trace.show()

# Top-k most active layers by L2 norm
trace.most_active(top_k=5)

# First layer where representation stabilizes
trace.stable_at(threshold=0.01)

# Overlay two traces on the same chart
trace_a.compare(trace_b)

# Raw activation tensors
trace.activations["encoder.layer.4"]   # → torch.Tensor
```

### `ProbeTrainer`

```python
trainer = ProbeTrainer(
    cv_folds=3,                          # default: 3-fold cross-validation
    max_iter=1000,                        # logistic regression iterations
    test_size=0.3,                        # fallback split when too few samples
    features=["mean", "std", "l2_norm"], # which stats to use as features
)
```

Available features: `mean`, `std`, `l2_norm`, `sparsity`, `min_val`, `max_val`

---

## Roadmap

### Phase 1 — Core Activation Analysis *(current)*
- [x] `HookEngine` — dynamic hook attachment/detach
- [x] `ActivationStore` — DuckDB-backed storage
- [x] `Axis` API — semantic comparison dimensions
- [x] `ProbeTrainer` — linear probing per layer
- [x] `ProbeResults` — terminal report + Plotly dashboards
- [x] `TraceResults` — single-input activation trajectory
- [x] Sentence comparison with divergence detection

### Phase 2 — Neuron & Head Attribution
- [x] `NeuronAttributor` — top-k neurons per concept group
- [x] `HeadAttributor` — attention head specialization map
- [ ] Dead neuron detection

### Phase 3 — Advanced Analysis
- [x] Concept interference detection (shared neurons across axes)
- [ ] Activation trajectory drift (pre/post fine-tuning)
- [ ] Concept representation evolution across layers

### Phase 4 — Diagnostics & Reporting
- [ ] Gradient monitoring (vanishing/exploding detection)
- [ ] Training failure detection
- [ ] HTML/PDF report export
- [ ] CLI interface (`nndbg` command)

---

## Development

```bash
# Run tests
poetry run pytest

# With coverage report
poetry run pytest --cov=nndbg

# Format code
poetry run black nndbg/
poetry run ruff check nndbg/
```

---

## License

MIT — see [LICENSE](LICENSE) for details.