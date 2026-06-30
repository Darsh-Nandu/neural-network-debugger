# NNDbg — a diagnostic toolkit for neural networks

[![CI](https://github.com/Darsh-Nandu/neural-network-debugger/actions/workflows/ci.yml/badge.svg)](https://github.com/Darsh-Nandu/neural-network-debugger/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)

NNDbg wraps a PyTorch or HuggingFace model in a single `Inspector` and
answers the questions people actually ask when interpreting a neural
network: where a concept is encoded, which inputs caused an output, what
each attention head is doing, which layer causally produces a behaviour,
and what features a layer's activations decompose into.

## Analysis planes

| `inspector.<plane>` | Answers | Method |
|---|---|---|
| `probing` | Where is concept X encoded? | cross-validated linear probes per layer |
| `attribution` | Which inputs caused this output? | saliency, Integrated Gradients, Grad-CAM |
| `attention` | What does each head attend to? | per-head heatmaps, rollout, entropy |
| `patching` | Which layers causally produce a behaviour? | activation patching / causal tracing |
| `sae` | What sparse features does a layer learn? | sparse autoencoder, train + decompose |
| `latent` | Where do activations sit in a compressed space? | VAE latent space + anomaly detection |

Every result is a plain dataclass with a `.plot()` method (matplotlib,
zero-config) and an optional `.plotly()` method if you have `plotly`
installed.

See [docs/ROADMAP.md](docs/ROADMAP.md) for what's coming next (geometry/CKA,
neuron-level analysis, concept erasure) — this release deliberately covers
six planes well rather than many partially.

## Installation

```bash
pip install nndbg

# with interactive Plotly figures
pip install nndbg[plotly]
```

## Quick start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from nndbg import Inspector

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
inspector = Inspector(model, tokenizer)

inspector.summary()          # model + available planes
inspector.layers()[:5]       # every layer name you can refer to

# Attribution — which input tokens drove the prediction?
input_ids = tokenizer("The capital of France is", return_tensors="pt").input_ids
inspector.attribution.saliency(input_ids).plot()

# Attention — what does head 0 of layer 0 attend to?
inspector.attention.heads(input_ids, layer=0).plot()

# Probing — is sentiment linearly decodable, and from which layer?
dataset = [
    (tokenizer(text, return_tensors="pt").input_ids, label)
    for text, label in [
        ("I love this movie", 1), ("I hate this movie", 0),
        ("This is wonderful", 1), ("This is terrible", 0),
    ]
]
inspector.probing.fit(dataset, concept="sentiment").plot()
```

Any plain `nn.Module` works too — `tokenizer` is optional and only needed
for token-level labeling.

### Activation patching / causal tracing

```python
clean = tokenizer("The Eiffel Tower is in the city of", return_tensors="pt").input_ids
corrupted = tokenizer("The Space Needle is in the city of", return_tensors="pt").input_ids

result = inspector.patching.causal_trace(
    clean, corrupted, layers=inspector.find_layers(r"h\.\d+$")
)
result.plot()  # (layer x position) logit-recovery heatmap
```

### Sparse autoencoders and VAE latent analysis

```python
dataset = [tokenizer(t, return_tensors="pt").input_ids for t in texts]

# Sparse, (closer-to-)monosemantic feature decomposition
inspector.sae.train(dataset, layer="transformer.h.6", n_features=512)
inspector.sae.decompose(dataset, layer="transformer.h.6").plot()

# Compressed latent space + reconstruction-error anomaly detection
inspector.latent.train(dataset, layer="transformer.h.6", latent_dim=2)
result = inspector.latent.encode(dataset, layer="transformer.h.6")
result.plot()
result.anomalies()  # indices of outlier examples
```

## Development

```bash
git clone https://github.com/Darsh-Nandu/neural-network-debugger
cd neural-network-debugger
pip install -e ".[dev]"
ruff check nndbg tests
pytest
```
