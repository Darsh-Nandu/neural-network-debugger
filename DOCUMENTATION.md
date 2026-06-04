# NNDbg — Professional Usage Documentation

**Version:** 0.1.0  
**Repository:** https://github.com/Darsh-Nandu/nndbg  
**License:** MIT

---

## Table of Contents

1. [Overview](#1-overview)
2. [Installation](#2-installation)
3. [Core Concepts](#3-core-concepts)
4. [Getting Started](#4-getting-started)
5. [The ModelProbe API](#5-the-modelprobe-api)
6. [Defining Axes](#6-defining-axes)
7. [Running Analysis](#7-running-analysis)
8. [Working with ProbeResults](#8-working-with-proberesults)
9. [Activation Trajectory (Trace)](#9-activation-trajectory-trace)
10. [Customizing the ProbeTrainer](#10-customizing-the-probetrainer)
11. [Persisting Results to Disk](#11-persisting-results-to-disk)
12. [Using Custom Models](#12-using-custom-models)
13. [Architecture Query](#13-architecture-query)
14. [Advanced Attribution (Phase 2)](#14-advanced-attribution-phase-2)
15. [Practical Use Cases](#15-practical-use-cases)
16. [Troubleshooting](#16-troubleshooting)
17. [API Quick Reference](#17-api-quick-reference)

---

## 1. Overview

NNDbg (Neural Network Debugger) is a Python library for **mechanistic interpretability** of transformer-based and other PyTorch neural networks. It helps you understand *where* inside a model specific concepts are encoded, by running controlled experiments and measuring concept separability at every layer.

The core idea is simple:

1. You define **axes** — dimensions of comparison (language, domain, sentiment, etc.)
2. Each axis has labeled **groups** of text samples (e.g., `english`, `french`, `hindi`)
3. NNDbg runs every sample through the model, captures activations at every layer
4. It trains a **linear probe** (logistic regression) per layer to classify which group a sample belongs to
5. The probe's **classification accuracy** is the "probe score" — a measure of how well that layer encodes the concept
6. You visualize the result as a heatmap, bar charts, or a terminal report

This technique is called **probing** or **diagnostic classification** in the interpretability literature.

---

## 2. Installation

### Requirements

| Requirement | Minimum Version |
|---|---|
| Python | 3.10 |
| PyTorch | 2.0.0 |
| CUDA (optional) | Any PyTorch-compatible version |

### Install from Source

```bash
git clone https://github.com/Darsh-Nandu/nndbg.git
cd nndbg
```

**Using Poetry (recommended):**
```bash
poetry install
```

**Using pip:**
```bash
pip install -e .
```

### Verify Installation

```python
import nndbg
print(nndbg.__version__)   # 0.1.0

from nndbg import ModelProbe, ProbeResults
```

### Optional: GPU Acceleration

NNDbg uses `nndbg.utils.get_device()` to automatically select the best available device. No configuration needed — it will use CUDA if available, MPS on Apple Silicon, or fall back to CPU.

---

## 3. Core Concepts

### 3.1 Hooks

NNDbg uses PyTorch's **forward hook** mechanism to capture model internals without modifying any model code. A `HookEngine` object:

- Iterates over every `named_module()` in the model
- Registers a `register_forward_hook` on each one
- During a forward pass, each hook captures the output tensor and stores it under the layer's name
- Hooks are cleanly removed after use via `detach()`

This is completely non-invasive — the model's behavior is unchanged.

### 3.2 Axes and Groups

An **axis** represents one dimension of semantic variation:

```
axis: "language"
  group "english"  → [text1, text2, text3, ...]
  group "french"   → [text1, text2, text3, ...]
  group "hindi"    → [text1, text2, text3, ...]
```

**Design rules for good axes:**

- Groups should differ along **exactly one dimension** — the one you're studying
- All other factors (topic, length, complexity) should be roughly balanced across groups
- The example texts in `test_probe.py` demonstrate this: all groups use the same underlying news sentences, just in different languages or domains
- Use at least **5–10 samples per group** for meaningful probe scores; more is better

### 3.3 Linear Probing

For each layer, NNDbg trains a `LogisticRegression` classifier using scikit-learn. The feature vector for each sample consists of 6 activation statistics: `mean`, `std`, `l2_norm`, `sparsity`, `min_val`, `max_val`.

Evaluation uses **k-fold cross-validation** (default k=3). When there are too few samples for cross-validation, it falls back to a train/test split.

**Interpreting probe scores:**

| Score range | Interpretation |
|---|---|
| 0.9–1.0 | Layer almost perfectly encodes the concept |
| 0.8–0.9 | Strong encoding |
| 0.6–0.8 | Moderate encoding, partial separation |
| 0.5–0.6 | Near chance — concept weakly or not encoded |
| < 0.5 | No encoding detected (unlikely given enough groups) |

For binary classification (2 groups), chance is 0.5. For N groups, chance is 1/N.

### 3.4 Activation Statistics

For each layer and each input sample, NNDbg reduces the full activation tensor to 6 scalar statistics:

| Statistic | Formula | Captures |
|---|---|---|
| `mean` | `tensor.mean()` | Average activation level |
| `std` | `tensor.std()` | Variability / spread |
| `l2_norm` | `‖tensor‖₂` | Overall magnitude |
| `sparsity` | `(tensor == 0).mean()` | Fraction of dead neurons |
| `min_val` | `tensor.min()` | Minimum activation |
| `max_val` | `tensor.max()` | Maximum activation |

These statistics are computed on a flattened version of the tensor. NaN and Inf values are replaced with 0.

---

## 4. Getting Started

### Minimal Working Example

```python
from nndbg import ModelProbe

# 1. Load model
probe = ModelProbe.from_pretrained("bert-base-multilingual-cased")

# 2. Add at least one axis with at least 2 groups
probe.add_axis("language", {
    "english": ["Hello world.", "The cat sat on the mat.", "Good morning."],
    "french":  ["Bonjour monde.", "Le chat était assis.", "Bonjour."],
})

# 3. Run
results = probe.run()

# 4. View results
results.summary()
```

This will print a color-coded terminal report showing which layers encode the language distinction.

### Minimum Data Requirements

- At least **2 groups** per axis (will raise `ValueError` otherwise)
- At least **1 sample** per group (will raise `ValueError` otherwise)
- At least **4 samples per group** is recommended for meaningful probe scores
- At least **6 samples per group** is recommended for cross-validation (default k=3 folds)

---

## 5. The ModelProbe API

### 5.1 Loading from HuggingFace

```python
probe = ModelProbe.from_pretrained(
    "google/mt5-small",      # any HuggingFace model identifier
    store_path=":memory:",   # ":memory:" for in-memory, or a path like "./run.duckdb"
    max_length=128,          # tokenizer max sequence length
)
```

`from_pretrained` internally calls:
```python
AutoTokenizer.from_pretrained(model_name)
AutoModel.from_pretrained(model_name)
```

It requires the `transformers` package. If not installed:
```
ImportError: transformers not installed. Run: pip install transformers
```

### 5.2 Loading a Custom Model

```python
probe = ModelProbe.from_model(
    model,                   # any nn.Module
    tokenizer=my_tokenizer,  # any tokenizer with __call__() → input_ids
    model_name="my-model",   # display name used in reports
    store_path=":memory:",
    max_length=128,
)
```

The tokenizer must return a dict of tensors when called like:
```python
tokenizer(text, return_tensors="pt", truncation=True, padding=True)
```

### 5.3 Direct Constructor

```python
import torch.nn as nn
from nndbg import ModelProbe

probe = ModelProbe(
    model=my_model,               # nn.Module
    tokenizer=my_tokenizer,
    model_name="custom",
    store_path=":memory:",
    max_length=128,
    probe_trainer=custom_trainer, # optional, ProbeTrainer instance
)
```

---

## 6. Defining Axes

### 6.1 Basic Axis

```python
probe.add_axis("sentiment", {
    "positive": [
        "I absolutely loved this product!",
        "Best purchase I've ever made.",
        "Highly recommended, exceeded expectations.",
    ],
    "negative": [
        "Terrible experience, would not recommend.",
        "Worst product I've ever bought.",
        "Complete waste of money.",
    ],
    "neutral": [
        "It arrived in the expected time.",
        "The product matches the description.",
        "Standard packaging, nothing special.",
    ],
})
```

### 6.2 Chaining Multiple Axes

The `add_axis()` method returns `self`, so calls can be chained:

```python
probe \
    .add_axis("language", {"english": [...], "french": [...], "hindi": [...]}) \
    .add_axis("domain",   {"legal":   [...], "medical": [...], "code": [...]}) \
    .add_axis("sentiment",{"positive":[...], "negative":[...]})
```

### 6.3 Axis Design Best Practices

**Balance your groups:**
```python
# Good: same topics, different language
"english": ["The stock market fell today.", "Scientists found a new species."]
"french":  ["Le marché boursier a chuté.", "Les scientifiques ont trouvé une espèce."]

# Bad: different topics AND different language — axis is confounded
"english": ["Stock markets are volatile."]
"french":  ["Les chats sont mignons."]
```

**Use semantically diverse samples within each group** — covering different topics, sentence lengths, and structures makes the probe more robust.

**Minimum samples per group:** 4 (for the fallback train/test split). Recommended: 10+.

### 6.4 Axis Validation

The `Axis` dataclass validates inputs on creation:

- Fewer than 2 groups → `ValueError`
- Empty sample list for any group → `ValueError`

---

## 7. Running Analysis

### 7.1 Basic Run

```python
results = probe.run()
```

`run()` executes the full pipeline:

1. Attaches hooks to all layers
2. Runs every sample through the model (with `torch.no_grad()`)
3. Collects activation statistics per layer
4. Persists stats to DuckDB
5. Trains linear probes per layer × axis
6. Detaches all hooks
7. Returns a `ProbeResults` object

Progress is shown via `tqdm` progress bars, one per (axis, group) pair.

### 7.2 What Happens Internally

```
probe.run()
  │
  ├── for each axis:
  │     for each group:
  │       for each text sample:
  │         ├── tokenize(text)
  │         ├── model(**inputs)   # forward pass with hooks active
  │         ├── for each layer:
  │         │     capture activation tensor
  │         │     compute [mean, std, l2_norm, sparsity, min, max]
  │         │     store to DuckDB
  │         └── store in-memory for probe training
  │
  └── for each axis:
        for each layer:
          train LogisticRegression on activation stats
          record probe accuracy → probe_scores[axis][layer]
```

### 7.3 Error Handling

If a forward pass raises an exception (e.g., model returns an unexpected output format), it is caught and logged as a warning:

```
WARNING: Forward pass issue: <error message>
```

The analysis continues. Layers with no captured data will have a probe score of 0.0.

---

## 8. Working with ProbeResults

### 8.1 Terminal Report

```python
results.summary()
# Prints a rich color-coded report to stdout

# Control how many layers to show per axis
results.summary(top_k=10)         # show top 10 layers per axis
results.summary(top_k=None)       # show ALL layers
results.summary(min_score=0.7)    # only show layers above 70% accuracy
results.summary(top_k=5, min_score=0.6)  # combined
```

The report shows a ranked table per axis with:
- Layer name
- Probe score (3 decimal places)
- Color-coded confidence bar (green ≥ 0.8, yellow ≥ 0.6, red < 0.6)

### 8.2 Interactive Dashboard

```python
results.show()
# Opens an interactive Plotly bar chart in your browser

results.show(top_k=20)            # show top 20 layers
results.show(min_score=0.65)      # only layers above 65%
```

The dashboard has one panel per axis. Bars are colored by score intensity. Hovering shows the full layer name and exact score.

If Plotly is not installed, `show()` falls back to `summary()`.

### 8.3 Layer × Axis Heatmap

```python
results.plot_heatmap()
# Opens a Plotly heatmap: rows = layers, columns = axes

results.plot_heatmap(min_score=0.7)  # only layers encoding at least one axis > 70%
```

The heatmap uses the Viridis colorscale. It's the most compact way to see the full encoding pattern across all layers and all axes simultaneously.

### 8.4 Querying Top Encoding Layers

```python
# Get the top-5 layers for "language" axis
layers = results.encoding_layers("language")
# → [("encoder.layer.4", 0.89), ("encoder.layer.3", 0.87), ...]

# All layers, no limit
layers = results.encoding_layers("language", top_k=None)

# All layers above 80% accuracy
layers = results.encoding_layers("language", top_k=None, min_score=0.8)

# Top 10 layers
layers = results.encoding_layers("language", top_k=10)
```

Returns a list of `(layer_name, score)` tuples, sorted best-first.

### 8.5 Exporting Results

```python
data = results.to_dict()
# Returns:
# {
#   "run_id":       "a1b2c3d4",
#   "model_name":   "bert-base-multilingual-cased",
#   "axes":         ["language", "domain"],
#   "probe_scores": {
#     "language": {"encoder.layer.0": 0.51, "encoder.layer.4": 0.89, ...},
#     "domain":   {"encoder.layer.0": 0.48, "encoder.layer.8": 0.93, ...},
#   }
# }

import json
with open("results.json", "w") as f:
    json.dump(data, f, indent=2)
```

---

## 9. Activation Trajectory (Trace)

The `trace()` method is independent of axes and probe training. It runs a single text through the model and captures activation statistics at every layer, letting you see how the representation evolves.

### 9.1 Basic Trace

```python
trace = probe.trace("Le chat était assis sur le tapis.")

# Terminal summary table
trace.summary()

# Interactive Plotly chart (mean, std, l2 norm per layer)
trace.show()
```

### 9.2 Key Queries

```python
# Top-5 most strongly activated layers (by L2 norm)
top = trace.most_active(top_k=5)
# → [("encoder.layer.11", 142.3), ("encoder.layer.10", 138.7), ...]

# Find where representation stabilizes
stable_layer = trace.stable_at(threshold=0.01)
# → "encoder.layer.8"
# Interpretation: after this layer, the mean activation changes by < 0.01 per layer
```

`stable_at()` iterates through layers in order and returns the first layer where the change in mean activation drops below `threshold`. This can indicate where the model has "finished processing" the main semantic content.

### 9.3 Accessing Raw Tensors

```python
# Get the full activation tensor for a specific layer
tensor = trace.activations["encoder.layer.4"]
# → torch.Tensor of shape [batch_size, seq_len, hidden_size]

print(tensor.shape)
```

Use this for custom analysis — e.g., applying your own classifiers, computing cosine similarity between specific layer activations, or visualizing with UMAP/t-SNE.

### 9.4 Comparing Two Inputs

```python
english = probe.trace("The cat sat on the mat.")
french  = probe.trace("Le chat était assis sur le tapis.")

# Overlay both activation trajectories
# Automatically marks the layer of maximum divergence
english.compare(french)
```

The comparison chart shows mean activation and L2 norm for both traces across all layers. A vertical dashed line marks the layer where the two representations diverge most strongly.

---

## 10. Customizing the ProbeTrainer

### 10.1 Default Configuration

The default `ProbeTrainer` uses:
- 3-fold cross-validation
- All 6 activation statistics as features
- 1000 logistic regression iterations
- 0.3 test size for the fallback split

### 10.2 Custom Configuration

```python
from nndbg.probing.trainer import ProbeTrainer

trainer = ProbeTrainer(
    cv_folds=5,                       # more folds = more reliable, but slower
    max_iter=2000,                    # more iterations = better convergence
    test_size=0.2,                    # fallback split ratio
    features=["mean", "std"],         # use only 2 features (faster, less info)
)
```

### 10.3 Available Features

| Feature | Meaning | When useful |
|---|---|---|
| `mean` | Average activation | Always useful |
| `std` | Standard deviation | Captures activation spread |
| `l2_norm` | Euclidean magnitude | Good proxy for "how active is this layer" |
| `sparsity` | Fraction of zero activations | Useful after ReLU activations |
| `min_val` | Minimum activation | Edge cases |
| `max_val` | Maximum activation | Peak activation behavior |

For speed, use `features=["mean", "std", "l2_norm"]`. For maximum information, use all 6 (default).

### 10.4 Passing to ModelProbe

```python
probe = ModelProbe.from_pretrained(
    "bert-base-multilingual-cased",
    probe_trainer=trainer,
)
```

Or via the constructor:
```python
probe = ModelProbe(model, tokenizer, probe_trainer=trainer)
```

---

## 11. Persisting Results to Disk

By default, NNDbg uses an **in-memory DuckDB database**. For large models, many axes, or repeated experiments, you can persist to disk:

```python
probe = ModelProbe.from_pretrained(
    "google/mt5-small",
    store_path="./experiment_run.duckdb",
)
```

The DuckDB file persists three tables:

| Table | Contents |
|---|---|
| `runs` | Run ID, model name, timestamp, config JSON |
| `activations` | Per-sample activation stats for every layer |
| `layer_stats` | Probe scores per layer per axis |

You can query these tables directly with DuckDB's Python API for custom analysis:

```python
import duckdb

conn = duckdb.connect("./experiment_run.duckdb")

# Query all probe scores for the "language" axis
rows = conn.execute("""
    SELECT layer_name, probe_score
    FROM layer_stats
    WHERE axis_name = 'language'
    ORDER BY probe_score DESC
""").fetchall()

for layer, score in rows:
    print(f"{layer}: {score:.3f}")
```

---

## 12. Using Custom Models

Any `nn.Module` that accepts tokenized inputs (a dict of tensors) and returns either a tensor or a tuple can be used with NNDbg.

### 12.1 Custom PyTorch Model

```python
import torch.nn as nn
from nndbg import ModelProbe

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(30522, 768)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(768, 8) for _ in range(6)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, input_ids, **kwargs):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        return x

model = MyModel()

# Use a standard HuggingFace tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

probe = ModelProbe.from_model(model, tokenizer=tokenizer, model_name="my-transformer")
probe.add_axis("language", {"english": [...], "french": [...]})
results = probe.run()
```

### 12.2 Models with Non-Standard Outputs

If your model returns a complex object (e.g., a named tuple, a dataclass), the `HookEngine` captures outputs at each *submodule* level, not just the final output. Layer-level hooks see the raw tensor output of each `nn.Module` sub-component, so complex final outputs are handled transparently.

---

## 13. Architecture Query

You can inspect the model's layer structure without running any analysis:

```python
arch = probe.architecture()
# Returns a dict:
# {
#   "encoder.layer.0": {
#       "type": "BertLayer",
#       "parameters": 7087872,
#       "trainable": 7087872,
#       "frozen": 0
#   },
#   ...
# }

# Count layers
print(f"Total layers: {len(arch)}")

# Filter to specific layer types
encoder_layers = {k: v for k, v in arch.items() if "attention" in k}
```

---

## 14. Advanced Attribution (Phase 2)

The `attribution` module is under development (Phase 2). The API is planned as follows:

### 14.1 Neuron Attribution

Identifies which *individual neurons* are most responsible for encoding a concept in a specific group:

```python
from nndbg.attribution import NeuronAttributor

attr = NeuronAttributor(probe)
attr.fit("language")                            # run attribution for an axis
attr.show("language", group="french", top_k=10) # top-10 neurons for "french"

# Exclusive neurons: fire for "french" but NOT for other groups
attr.show("language", group="french", top_k=10, mode="exclusive")
```

### 14.2 Head Attribution

For transformer models, identifies which attention heads specialize in a concept:

```python
from nndbg.attribution import HeadAttributor

heads = HeadAttributor(probe)
heads.fit("language")
heads.show("language", top_k=10)
heads.plot_heatmap("language")     # head × layer heatmap
```

### 14.3 Interference Detection

Finds neurons shared across two axes — neurons that encode *both* language and domain simultaneously:

```python
from nndbg.attribution import InterferenceDetector

detector = InterferenceDetector(attr)
detector.check("language", "domain")   # prints shared neurons
```

*Note: These APIs are stubbed out but not fully implemented in v0.1.0.*

---

## 15. Practical Use Cases

### Use Case 1: Finding Where Language Separates in mBERT

```python
from nndbg import ModelProbe

probe = ModelProbe.from_pretrained("bert-base-multilingual-cased")

probe.add_axis("language", {
    "english": ["The stock market crashed today."] * 10,
    "french":  ["Le marché boursier s'est effondré."] * 10,
    "hindi":   ["आज शेयर बाजार गिर गया।"] * 10,
    "spanish": ["El mercado bursátil se derrumbó hoy."] * 10,
})

results = probe.run()
results.summary(top_k=5)

# Find the first layer where language is strongly encoded (> 80%)
early_layers = results.encoding_layers("language", top_k=None, min_score=0.8)
print("Language encoded strongly from:", early_layers[-1][0] if early_layers else "none")
```

### Use Case 2: Comparing Domain Encoding Before and After Fine-Tuning

```python
# Before fine-tuning
probe_base = ModelProbe.from_pretrained("bert-base-uncased", store_path="base.duckdb")
probe_base.add_axis("domain", {"legal": [...], "medical": [...], "code": [...]})
results_base = probe_base.run()

# After fine-tuning
probe_ft = ModelProbe.from_model(finetuned_model, tokenizer, store_path="ft.duckdb")
probe_ft.add_axis("domain", {"legal": [...], "medical": [...], "code": [...]})
results_ft = probe_ft.run()

# Compare
print("Base — top domain layers:")
for layer, score in results_base.encoding_layers("domain", top_k=5):
    print(f"  {layer}: {score:.3f}")

print("Fine-tuned — top domain layers:")
for layer, score in results_ft.encoding_layers("domain", top_k=5):
    print(f"  {layer}: {score:.3f}")
```

### Use Case 3: Tracing a Specific Sentence

```python
probe = ModelProbe.from_pretrained("bert-base-multilingual-cased")

trace = probe.trace("The defendant shall appear before the court.")
trace.summary()

print("Most active layers:", trace.most_active(top_k=3))
print("Representation stabilizes at:", trace.stable_at())

# Compare legal vs medical language
legal = probe.trace("The defendant shall appear before the court.")
medical = probe.trace("Patient presents with acute symptoms.")
legal.compare(medical)
```

### Use Case 4: Jupyter Notebook Workflow

```python
# In a Jupyter notebook cell:

from nndbg import ModelProbe
import pandas as pd

probe = ModelProbe.from_pretrained("distilbert-base-multilingual-cased")
probe.add_axis("sentiment", {
    "positive": positive_samples,
    "negative": negative_samples,
})
results = probe.run()

# Export to DataFrame for custom analysis
data = results.to_dict()
scores = data["probe_scores"]["sentiment"]
df = pd.DataFrame(list(scores.items()), columns=["layer", "probe_score"])
df = df.sort_values("probe_score", ascending=False)

# Plotly heatmap inline
results.plot_heatmap()

# Interactive bar chart inline
results.show(top_k=15)
```

---

## 16. Troubleshooting

### `ValueError: No axes added`

You must call `probe.add_axis()` at least once before calling `probe.run()`.

```python
probe.add_axis("language", {...})
results = probe.run()           # now works
```

### `ValueError: Axis needs at least 2 groups`

Each axis must have at least 2 groups to be a meaningful comparison:

```python
# Wrong
probe.add_axis("language", {"english": [...]})  # only 1 group

# Right
probe.add_axis("language", {"english": [...], "french": [...]})
```

### `ImportError: transformers not installed`

Install the transformers library:
```bash
pip install transformers
```

### `Warning: Too few samples for reliable probe evaluation`

If a group has fewer than 4 samples, the probe score will be unreliable (returns 0.0). Add more samples:

```python
# Recommended: 10+ samples per group for reliable scores
probe.add_axis("language", {
    "english": [text1, text2, ..., text10],  # 10 samples minimum recommended
    "french":  [text1, text2, ..., text10],
})
```

### Forward pass warnings

```
WARNING: Forward pass issue: <error>
```

Some models return non-standard outputs from certain submodules. The warning is informational — the analysis continues. If many layers show this, check that your tokenizer output format is compatible with your model.

### Low probe scores across all layers

Possible causes:
- **Too few samples**: Increase to 10+ per group
- **Confounded axis**: Groups differ on multiple dimensions simultaneously — isolate one dimension
- **Model doesn't encode the concept**: Some models genuinely don't encode certain distinctions internally
- **Feature subset too narrow**: Try using all 6 default features

### Memory issues with large models

For models with many parameters (e.g., GPT-2, LLaMA):
- Reduce `max_length` (e.g., `max_length=64`)
- Use fewer samples per group
- Use a smaller model variant
- Run on GPU if available (NNDbg auto-selects CUDA)

---

## 17. API Quick Reference

### `ModelProbe`

| Method / Constructor | Signature | Returns |
|---|---|---|
| `from_pretrained` | `(model_name, store_path, max_length, **kwargs)` | `ModelProbe` |
| `from_model` | `(model, tokenizer, model_name, **kwargs)` | `ModelProbe` |
| `__init__` | `(model, tokenizer, model_name, store_path, max_length, probe_trainer)` | `ModelProbe` |
| `add_axis` | `(name: str, groups: Dict[str, List[str]])` | `self` (chainable) |
| `run` | `()` | `ProbeResults` |
| `trace` | `(text: str)` | `TraceResults` |
| `architecture` | `()` | `Dict[str, Dict]` |

### `ProbeResults`

| Method | Signature | Returns |
|---|---|---|
| `summary` | `(top_k=5, min_score=0.0)` | `str` (also prints) |
| `show` | `(top_k=None, min_score=0.0)` | `None` (opens Plotly) |
| `plot_heatmap` | `(min_score=0.0)` | `None` (opens Plotly) |
| `encoding_layers` | `(axis_name, top_k=5, min_score=0.0)` | `List[Tuple[str, float]]` |
| `to_dict` | `()` | `dict` |

### `TraceResults`

| Method / Attribute | Signature | Returns |
|---|---|---|
| `summary` | `()` | `str` (also prints) |
| `show` | `()` | `None` (opens Plotly) |
| `compare` | `(other: TraceResults)` | `None` (opens Plotly) |
| `most_active` | `(top_k=5)` | `List[Tuple[str, float]]` |
| `stable_at` | `(threshold=0.01)` | `Optional[str]` |
| `activations` | attribute | `Dict[str, torch.Tensor]` |
| `layers` | attribute | `List[LayerTrace]` |
| `text` | attribute | `str` |

### `ProbeTrainer`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `cv_folds` | `int` | `3` | Number of cross-validation folds |
| `max_iter` | `int` | `1000` | Max iterations for logistic regression |
| `test_size` | `float` | `0.3` | Fallback train/test split ratio |
| `features` | `List[str]` | all 6 | Activation statistics to use as features |

### `Axis`

| Attribute / Property | Description |
|---|---|
| `name` | Axis name (e.g., `"language"`) |
| `groups` | `Dict[str, List[str]]` — group labels to sample lists |
| `group_names` | Property — list of group label strings |
| `total_samples` | Property — total sample count across all groups |

---

*NNDbg v0.1.0 — MIT License*
