"""
Vendi Reduction Trajectory Script
==================================
Fits a BERTopic model on 20 Newsgroups (P1 config: min_cluster_size=15, seed=42),
then runs Vendi topic reduction step-by-step from the initial k down to 1 topic,
recording VS2, NPMI, C_v, and TD (Topic Diversity = word_uniqueness_10) at every step.

Output:
  - experiments/results/vendi_trajectory/trajectory.csv
  - experiments/results/vendi_trajectory/trajectory_metadata.json

The CSV is designed to be fed directly to an LLM to generate plotting code.
"""

import sys
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.utils.data_loaders import prepare_dataset, print_dataset_stats
from experiments.utils.models import create_bertopic_model, ModelConfig
from experiments.utils.metrics import (
    compute_coherence_cv,
    compute_coherence_npmi,
    compute_word_uniqueness,
    compute_vendi_diversity,
)

# ─── Configuration ────────────────────────────────────────────────────────────
SEED            = 42
MIN_CLUSTER_SIZE = 15
SAVE_DIR        = "experiments/results/vendi_trajectory"
# ──────────────────────────────────────────────────────────────────────────────


def measure_at_k(model, docs):
    """Compute the four metrics for the current model state."""
    vs2   = compute_vendi_diversity(model, q=2.0)
    npmi  = compute_coherence_npmi(model, docs)
    cv    = compute_coherence_cv(model, docs)
    td    = compute_word_uniqueness(model, top_n=10)
    k     = len(set(model.topics_) - {-1})
    return {"k": k, "VS2": vs2, "NPMI": npmi, "C_v": cv, "TD": td}


def main():
    save_path = Path(SAVE_DIR)
    save_path.mkdir(parents=True, exist_ok=True)

    # ── 1. Load dataset ────────────────────────────────────────────────────────
    print("=" * 70)
    print("Vendi Reduction Trajectory  |  20 Newsgroups")
    print("=" * 70)
    print("\n[1/3] Loading dataset & embeddings...")
    dataset = prepare_dataset(name="20newsgroups")
    print_dataset_stats(dataset)

    # ── 2. Fit base model (P1 config) ─────────────────────────────────────────
    print("\n[2/3] Fitting base BERTopic model (Vendi, mcs=15, seed=42)...")
    config = ModelConfig(
        name="vendi_trajectory_base",
        reduction_method="vendi",
        min_cluster_size=MIN_CLUSTER_SIZE,
        seed=SEED,
    )
    model = create_bertopic_model(config)
    t0 = time.time()
    topics, _ = model.fit_transform(dataset.docs, dataset.embeddings)
    fit_time = time.time() - t0
    initial_k = len(set(topics) - {-1})
    print(f"  ✓ Fit complete in {fit_time:.1f}s  |  initial k = {initial_k}")

    # ── 3. Reduction trajectory ────────────────────────────────────────────────
    print(f"\n[3/3] Running reduction trajectory: k={initial_k} → 1")
    print("  (measuring VS2, NPMI, C_v, TD at every step)\n")

    records = []

    # Measure at initial k
    print(f"  k={initial_k:>4}  (initial)", end="", flush=True)
    row = measure_at_k(model, dataset.docs)
    records.append(row)
    print(f"  VS2={row['VS2']:.3f}  NPMI={row['NPMI']:.4f}  C_v={row['C_v']:.4f}  TD={row['TD']:.4f}")

    # Reduce one topic at a time until k=1
    current_k = initial_k
    step = 0
    while current_k > 1:
        target_k = current_k - 1
        step += 1

        t_step = time.time()
        model.reduce_topics(dataset.docs, nr_topics=target_k, use_ctfidf=True)
        step_time = time.time() - t_step

        current_k = len(set(model.topics_) - {-1})

        print(f"  k={current_k:>4}  (step {step:>4})  [{step_time:.1f}s]", end="", flush=True)
        row = measure_at_k(model, dataset.docs)
        records.append(row)
        print(f"  VS2={row['VS2']:.3f}  NPMI={row['NPMI']:.4f}  C_v={row['C_v']:.4f}  TD={row['TD']:.4f}")

        # Safety: if reduce_topics didn't actually reduce, break
        if current_k >= target_k + 1 and step > 1:
            print(f"  ⚠ Reduction stalled at k={current_k}, stopping.")
            break

    # ── 4. Save results ────────────────────────────────────────────────────────
    df = pd.DataFrame(records)

    csv_path = save_path / "trajectory.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Trajectory CSV saved → {csv_path}")
    print(df.to_string(index=False))

    # Metadata JSON (for LLM context)
    metadata = {
        "description": (
            "Vendi topic reduction trajectory on 20 Newsgroups. "
            "Starting from the BERTopic base model (P1 config), topics are merged "
            "one-by-one using Vendi (VS2) reduction until k=1. "
            "Metrics are recorded at every step."
        ),
        "dataset": "20newsgroups",
        "seed": SEED,
        "min_cluster_size": MIN_CLUSTER_SIZE,
        "reduction_method": "vendi (ctfidf embeddings)",
        "initial_k": initial_k,
        "final_k": int(df["k"].min()),
        "n_steps": len(df),
        "columns": {
            "k":    "Number of topics remaining after this merge step",
            "VS2":  "Vendi Score (q=2) — diversity of topic embeddings; higher = more diverse",
            "NPMI": "C_NPMI coherence — semantic quality of topics; higher = more coherent",
            "C_v":  "C_v coherence — semantic quality of topics; higher = more coherent",
            "TD":   "Topic Diversity (word uniqueness @10) — lexical diversity; higher = more unique words across topics",
        },
        "plot_goal": (
            "3 mini scatter/line plots: "
            "(1) VS2 vs NPMI, (2) VS2 vs C_v, (3) VS2 vs TD. "
            "x-axis = VS2, y-axis = each metric. "
            "Color or annotate points by k value."
        ),
        "csv_path": str(csv_path),
    }

    meta_path = save_path / "trajectory_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata JSON saved  → {meta_path}")

    print("\n" + "=" * 70)
    print("Done! Feed trajectory.csv + trajectory_metadata.json to an LLM")
    print("to generate plotting code in a Jupyter notebook.")
    print("=" * 70)


if __name__ == "__main__":
    main()
