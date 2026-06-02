# NNDbg — Neural Network Semantic Activation Analyzer

> Deep activation analysis for neural networks. Feed any model contrasting inputs — languages, domains, sentiments — and discover *where* in the model those concepts are encoded.

---

## What It Does

NNDbg answers questions like:

- **Which layer encodes language identity?** (English vs French vs Hindi)
- **Where does domain knowledge live?** (legal vs medical vs code)
- **How does fine-tuning change the model internally?**
- **Which neurons fire exclusively for sentiment?**

It does this by attaching hooks to any PyTorch model, capturing activations across every layer, and training linear probes to measure concept encoding — all without modifying the model source code.

---

## Quick Start

```python
from nndbg import ModelProbe

# Load any HuggingFace model
probe = ModelProbe.from_pretrained("google/mt5-small")

# Add comparison axes
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

probe.add_axis("domain", {
    "legal":   ["The defendant shall appear before the court.", ...],
    "medical": ["Patient presents with acute symptoms.", ...],
    "code":    ["def forward(self, x): return self.layer(x)", ...],
})

# Run analysis
results = probe.run()

# View results
print(results.summary())
results.show()          # Interactive Plotly dashboard
results.plot_heatmap()  # Layer × Axis encoding map
```

---

## Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/nndbg.git
cd nndbg

# Install with Poetry
poetry install

# Or with pip
pip install -e .
```

**Requirements:** Python ≥ 3.10, PyTorch ≥ 2.0

---

## Core Concepts

### Axes
An **axis** is a comparison dimension. You define groups of text samples that differ along one dimension (language, topic, sentiment, etc.).

```python
probe.add_axis("sentiment", {
    "positive": ["Great product!", "Absolutely loved it.", ...],
    "negative": ["Terrible experience.", "Worst purchase ever.", ...],
    "neutral":  ["It arrived on time.", "Standard packaging.", ...],
})
```

### Linear Probing
For each layer, NNDbg trains a **linear probe** (logistic regression) on the captured activations. The probe's accuracy measures how much that layer's activations discriminate between groups.

- High accuracy → layer encodes the concept
- Low accuracy → layer doesn't encode the concept (or encodes it implicitly)

### Probe Scores
The output is a matrix of **probe scores** per layer per axis:

```
               language  domain  sentiment
encoder.0        0.51     0.48     0.52
encoder.4        0.89     0.61     0.58   ← language crystallizes here
encoder.8        0.87     0.93     0.64   ← domain separates here
encoder.12       0.82     0.88     0.91   ← sentiment late
```

---

## API Reference

### `ModelProbe`

```python
# From HuggingFace
probe = ModelProbe.from_pretrained("bert-base-multilingual-cased")

# From existing model
probe = ModelProbe.from_model(my_model, tokenizer=my_tokenizer)

# Options
probe = ModelProbe.from_pretrained(
    "google/mt5-small",
    store_path="./my_run.duckdb",  # persist to disk
    max_length=256,
    batch_size=16,
)
```

### `probe.add_axis(name, groups)`
Add a comparison axis. Chainable.

```python
probe.add_axis("language", {...}).add_axis("domain", {...})
```

### `probe.run()` → `ProbeResults`
Run the full analysis pipeline.

### `ProbeResults`

```python
results.summary()                        # Text report
results.show()                           # Interactive dashboard
results.plot_heatmap()                   # Layer × Axis heatmap
results.encoding_layers("language", k=5) # Top-5 encoding layers
results.to_dict()                        # Export as dict
```

---

## Architecture

```
nndbg/
├── probe.py          # ModelProbe — main API
├── results.py        # ProbeResults — output + visualization
├── hooks/
│   ├── engine.py     # HookEngine — dynamic hook attachment
│   └── registry.py  # HookRegistry — layer metadata
├── storage/
│   └── store.py      # ActivationStore — DuckDB backend
├── probing/
│   ├── axis.py       # Axis dataclass
│   └── trainer.py    # ProbeTrainer — linear probes
├── attribution/      # (Phase 2) neuron + head attribution
├── visualization/    # (Phase 2) advanced dashboards
└── utils/
    ├── logging.py
    └── device.py
```

---

## Roadmap

### Phase 1 (Current) — Core Activation Analysis
- [x] HookEngine — dynamic hook attachment
- [x] ActivationStore — DuckDB-backed storage
- [x] Axis API — semantic comparison dimensions
- [x] Linear probing — per-layer concept encoding scores
- [x] ProbeResults — visualization with Plotly

### Phase 2 — Neuron & Head Attribution
- [ ] NeuronAttributor — top-k neurons per concept
- [ ] HeadAttributor — attention head specialization map
- [ ] Dead neuron detection

### Phase 3 — Advanced Analysis
- [ ] Concept interference detection (shared neurons across axes)
- [ ] Activation trajectory (how representations evolve layer by layer)
- [ ] Pre/post fine-tune drift analysis

### Phase 4 — Diagnostics
- [ ] Gradient monitoring (vanishing/exploding)
- [ ] Training failure detection
- [ ] HTML/PDF report export

---

## Development

```bash
# Run tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=nndbg

# Format
poetry run black nndbg/
poetry run ruff check nndbg/
```

---

## License

MIT
