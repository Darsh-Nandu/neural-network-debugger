"""Worked example: where does GPT-2 small represent sentiment?

Runs three planes on real data and writes the numbers to examples/results/:
  1. probing   - sentiment accuracy of a linear probe at every block
  2. erasure   - INLP on the best layer, to check the signal is linear
  3. patching  - causal trace of a factual prompt

    pip install -e . datasets
    python examples/gpt2_sentiment.py
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from nndbg import Inspector

OUT = Path(__file__).parent / "results"
CLEAN = "The Brandenburg Gate is in the city of"
CORRUPTED = "The Golden Gate Bridge is in the city of"  # same token count as CLEAN


def sst2_sample(tokenizer, n_per_class: int, seed: int):
    """Balanced, seeded sample of SST-2 training sentences as (input_ids, label)."""
    rows = load_dataset("nyu-mll/glue", "sst2", split="train")
    rng = random.Random(seed)
    by_label: dict[int, list[str]] = {0: [], 1: []}
    for i in rng.sample(range(len(rows)), len(rows)):
        row = rows[i]
        # skip fragments, SST-2 has many two-word phrases
        if len(row["sentence"].split()) >= 6 and len(by_label[row["label"]]) < n_per_class:
            by_label[row["label"]].append(row["sentence"].strip())
        if all(len(v) == n_per_class for v in by_label.values()):
            break
    pairs = [(text, label) for label, texts in by_label.items() for text in texts]
    rng.shuffle(pairs)
    return [
        (tokenizer(text, return_tensors="pt", truncation=True, max_length=64).input_ids, label)
        for text, label in pairs
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-class", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    OUT.mkdir(exist_ok=True)
    started = time.time()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
    inspector = Inspector(model, tokenizer)
    blocks = inspector.find_layers(r"h\.\d+$")

    dataset = sst2_sample(tokenizer, args.n_per_class, args.seed)

    # 1. probing, last token vs mean pooling
    probes = {}
    for pooling in ("last", "mean"):
        result = inspector.probing.fit(
            dataset, concept="sentiment", layers=blocks, pooling=pooling, max_iter=2000
        )
        probes[pooling] = result
        print(f"probe ({pooling}): best {result.best_layer()}", flush=True)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    for pooling, result in probes.items():
        ax.errorbar(range(len(blocks)), result.accuracy, yerr=result.std, marker="o", label=pooling)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("GPT-2 block")
    ax.set_ylabel("probe accuracy (5-fold CV)")
    ax.set_title("Linear probe for sentiment, SST-2")
    ax.legend(title="pooling")
    fig.tight_layout()
    fig.savefig(OUT / "probe_sentiment.png", dpi=150)
    plt.close(fig)

    # 2. erase the best direction(s) and watch the probe fall
    best = probes["mean"].best_layer()
    erased = inspector.erasure.inlp(
        dataset, concept="sentiment", layer=best, pooling="mean", n_iters=20, max_iter=2000
    )
    print(f"inlp on {best}: {erased.n_iters} iterations", flush=True)

    # 3. causal trace on a factual prompt
    clean = tokenizer(CLEAN, return_tensors="pt").input_ids
    corrupted = tokenizer(CORRUPTED, return_tensors="pt").input_ids
    trace = inspector.patching.causal_trace(clean, corrupted, layers=blocks)
    layer, position, value = trace.best_cell()
    trace.plot()
    plt.gcf().tight_layout()
    plt.gcf().savefig(OUT / "causal_trace.png", dpi=150)

    summary = {
        "model": "gpt2",
        "dataset": "nyu-mll/glue sst2 train, sentences with 6+ words",
        "n_samples": len(dataset),
        "seed": args.seed,
        "probe": {
            pooling: {
                "best_layer": r.best_layer(),
                "accuracy": dict(zip(r.layers, [round(a, 4) for a in r.accuracy])),
                "std": dict(zip(r.layers, [round(s, 4) for s in r.std])),
            }
            for pooling, r in probes.items()
        },
        "inlp": {
            "layer": best,
            "iterations": erased.n_iters,
            "train_accuracy_trace": [round(a, 4) for a in erased.accuracy_trace],
        },
        "causal_trace": {
            "clean": CLEAN,
            "corrupted": CORRUPTED,
            "target_token": tokenizer.decode([trace.target]),
            "best_layer": layer,
            "best_position": position,
            "best_token": trace.tokens[position] if trace.tokens else None,
            "best_value": round(float(value), 4),
        },
        "wall_time_sec": round(time.time() - started, 1),
        "device": str(inspector.device),
    }
    (OUT / "gpt2_sentiment.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
