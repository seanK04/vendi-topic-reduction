# Vendi Clustering for Topic Reduction

Code for the paper introducing Vendi Clustering (VC), a topic reduction method that selects each merge to minimize the loss in the Vendi Score, a spectral measure of diversity. This repository is a fork of [BERTopic](https://github.com/MaartenGr/BERTopic) with VC integrated into the topic reduction stage.

## Method code

- `bertopic/_vendi_reduction.py`: optimized VC at q = 2, using the Frobenius norm shortcut with cached row-square sums and rank-one Gram updates (O(1) per candidate merge, O(n^3) total)
- `bertopic/_vendi_reduction_general.py`: eigendecomposition-based VC for arbitrary q, used by the q-sensitivity ablation
- `bertopic/_llm_reduction.py`: iterative LLM-assisted merging baseline (Janssens et al., 2025)

Usage with BERTopic:

```python
from bertopic import BERTopic

topic_model = BERTopic(reduction_method="vendi", nr_topics=50)
topics, probs = topic_model.fit_transform(docs)
```

## Reproducing the paper

All experiments start from a BERTopic base model (all-MiniLM-L6-v2, UMAP, HDBSCAN) on 20 Newsgroups and AG News, seeds 42 to 44. Configs live in `experiments/configs/`, results are written to `experiments/results/`.

| Paper | Protocol | How to run |
|---|---|---|
| Table 1: VC vs. agglomerative | P1 | `python experiments/vendi_experiments.py --config experiments/configs/p1_20NG.yaml` |
| Table 1: VC vs. LLM (Gemma) | P2 | `python experiments/run_p2_with_llm.py --config experiments/configs/p2_20NG.yaml` (needs a vLLM server, see `slurm_p2.sh`) |
| Table 1: VC vs. LLM (GPT-4o-mini) | P2 | `python experiments/run_p2_with_openai.py --config experiments/configs/p2_20NG_openai.yaml` |
| Table 2 and Fig. 2: VC vs. direct HDBSCAN | P3 | `python experiments/vendi_experiments.py --config experiments/configs/p3_20NG.yaml` |
| Appendix: q-sensitivity ablation | P6 | `python experiments/vendi_experiments.py --config experiments/configs/p6_20NG.yaml` |
| Fig. 1: reduction trajectory | - | `python experiments/vendi_trajectory.py` |

Swap `20NG` for `AG` in any config name to run on AG News. Protocol 4 (robustness across HDBSCAN configurations) is supplementary and not reported in the paper.

## Setup

```bash
pip install -e .
pip install sentence-transformers umap-learn hdbscan gensim pyyaml tqdm openai
```

For the LLM experiments:

- GPT-4o-mini: set the `OPENAI_API_KEY` environment variable
- Gemma-3-12b-it: create a `.env` file in the repo root containing `HF_TOKEN=...` (gitignored), then submit `slurm_p2.sh` or start a vLLM server manually on port 8000

## Upstream

BERTopic is by Maarten Grootendorst ([docs](https://maartengr.github.io/BERTopic/), [paper](https://arxiv.org/abs/2203.05794)) and is MIT licensed. All modifications for Vendi Clustering are listed above; the rest of the library is unchanged.
