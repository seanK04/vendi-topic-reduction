"""
General Vendi Score-based Topic Reduction for arbitrary q.

This module implements the Vendi Score of order q for iterative topic merging,
using eigendecomposition. It supports any q > 0, including q=1 (Shannon entropy)
and q=inf. For q=2, prefer VendiReducer in _vendi_reduction.py which uses
the O(1) Frobenius norm shortcut.
"""

import numpy as np
from typing import Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import logsumexp
from tqdm import tqdm


class GeneralVendiReducer:
    """Vendi Score-based topic reduction for arbitrary order q.

    Uses eigendecomposition of the normalized similarity matrix to compute
    VS_q. For each candidate merge, builds the merged matrix and evaluates
    VS_q via eigenvalues. Complexity is O(m^5) total but tractable for
    typical topic counts (m <= 200).
    """

    def __init__(self, q: float = 1.0, verbose: bool = False):
        if q <= 0:
            raise ValueError(f"q must be > 0, got {q}")
        self.q = q
        self.verbose = verbose

    def compute_vendi_q_score(self, K: np.ndarray) -> float:
        """Compute VS_q from similarity matrix K using eigendecomposition.

        Args:
            K: Similarity matrix (m x m) with unit diagonal.

        Returns:
            Vendi Score of order q.
        """
        n = K.shape[0]
        eigenvalues = np.linalg.eigvalsh(K / n)
        eigenvalues = np.maximum(eigenvalues, 0.0)

        q = self.q

        if q == float("inf"):
            lam_max = np.max(eigenvalues)
            return 1.0 / lam_max if lam_max > 0 else 0.0

        if q == 1.0:
            # Shannon entropy: VS_1 = exp(-sum(lambda * log(lambda)))
            pos = eigenvalues[eigenvalues > 1e-30]
            return float(np.exp(-np.sum(pos * np.log(pos))))

        # General q: VS_q = (sum(lambda^q))^(1/(1-q))
        # Use log-space for numerical stability with large q
        pos = eigenvalues[eigenvalues > 1e-30]
        if len(pos) == 0:
            return 0.0
        log_terms = q * np.log(pos)
        log_sum = logsumexp(log_terms)
        return float(np.exp(log_sum / (1.0 - q)))

    def _build_merged_matrix(
        self, K: np.ndarray, a: int, b: int, n_a: float, n_b: float
    ) -> np.ndarray:
        """Construct the (m-1)x(m-1) similarity matrix after merging topics a and b.

        Args:
            K: Current similarity matrix (m x m).
            a, b: Indices of topics to merge.
            n_a, n_b: Document counts for topics a and b.

        Returns:
            Merged similarity matrix of size (m-1) x (m-1).
        """
        m = K.shape[0]
        K_ab = K[a, b]
        c = np.sqrt(n_a**2 + n_b**2 + 2 * n_a * n_b * K_ab)

        # New similarity row via weighted combination
        k_u = (n_a * K[a] + n_b * K[b]) / c
        k_u[a] = 1.0  # self-similarity

        # Remove row/col b, replace row/col a with merged
        mask = np.ones(m, dtype=bool)
        mask[b] = False

        K_new = K[np.ix_(mask, mask)]
        # After masking, index a stays at a if a < b, else shifts to a-1
        new_a = a if a < b else a - 1
        K_new[new_a, :] = k_u[mask]
        K_new[:, new_a] = k_u[mask]

        return K_new

    def find_best_merge(
        self, K: np.ndarray, n_vec: np.ndarray, current_score: float
    ):
        """Find the merge pair that maximizes the Vendi Score.

        Args:
            K: Current similarity matrix (m x m).
            n_vec: Array of document counts per topic.
            current_score: Current VS_q value.

        Returns:
            (best_delta, idx_a, idx_b): Best score change and merge indices.
        """
        m = K.shape[0]
        best_delta = -np.inf
        best_a, best_b = 0, 1

        for a in range(m):
            for b in range(a + 1, m):
                K_merged = self._build_merged_matrix(K, a, b, n_vec[a], n_vec[b])
                new_score = self.compute_vendi_q_score(K_merged)
                delta = new_score - current_score
                if delta > best_delta:
                    best_delta = delta
                    best_a, best_b = a, b

        return best_delta, best_a, best_b

    def apply_merge(
        self,
        K: np.ndarray,
        idx_a: int,
        idx_b: int,
        n_a: float,
        n_b: float,
        topic_embeddings: np.ndarray,
        active_topics: list,
        topic_sizes: dict,
    ):
        """Apply a merge and return updated state.

        Args:
            K: Current similarity matrix.
            idx_a, idx_b: Indices of topics to merge.
            n_a, n_b: Document counts.
            topic_embeddings: Current topic embedding matrix.
            active_topics: List of active topic IDs.
            topic_sizes: Dict mapping topic ID to document count.

        Returns:
            (K_new, active_topics_new, topic_embeddings_new)
        """
        K_new = self._build_merged_matrix(K, idx_a, idx_b, n_a, n_b)

        # Update embeddings (weighted centroid, unnormalized for storage)
        topic_embeddings[idx_a] = (
            n_a * topic_embeddings[idx_a] + n_b * topic_embeddings[idx_b]
        ) / (n_a + n_b)
        topic_sizes[active_topics[idx_a]] = n_a + n_b

        topic_embeddings_new = np.delete(topic_embeddings, idx_b, axis=0)
        active_topics_new = [t for i, t in enumerate(active_topics) if i != idx_b]

        return K_new, active_topics_new, topic_embeddings_new

    def reduce(
        self,
        embeddings: np.ndarray,
        topic_sizes: Dict[int, int],
        target_k: Optional[int] = None,
    ) -> Dict[int, int]:
        """Reduce topics by iteratively merging pairs that maximize VS_q.

        Args:
            embeddings: Topic embedding matrix (indexed by topic ID).
            topic_sizes: Dict mapping topic ID to document count.
            target_k: Target number of topics.

        Returns:
            Cumulative mapping from original topic IDs to reduced topic IDs.
        """
        active_topics = sorted(topic_sizes.keys())
        m = len(active_topics)

        topic_embeddings = np.array([embeddings[t] for t in active_topics])
        K = cosine_similarity(topic_embeddings)

        current_score = self.compute_vendi_q_score(K)

        if self.verbose:
            q_label = "∞" if self.q == float("inf") else self.q
            print(f"Initial: {m} topics, VS_q={current_score:.4f} (q={q_label})")

        cumulative_mapping = {t: t for t in active_topics}

        iteration = 0
        total_merges = m - target_k if target_k is not None else m - 1

        q_label = "∞" if self.q == float("inf") else self.q
        with tqdm(
            total=total_merges,
            desc=f"    Vendi reduction (q={q_label})",
            ncols=80,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n}/{total} [{elapsed}<{remaining}]",
            disable=not target_k,
        ) as pbar:
            while True:
                m = K.shape[0]

                if target_k is not None and m <= target_k:
                    break
                if m <= 1:
                    break

                n_vec = np.array([topic_sizes[t] for t in active_topics])
                best_delta, idx_a, idx_b = self.find_best_merge(
                    K, n_vec, current_score
                )

                a = active_topics[idx_a]
                b = active_topics[idx_b]

                K, active_topics, topic_embeddings = self.apply_merge(
                    K,
                    idx_a,
                    idx_b,
                    topic_sizes[a],
                    topic_sizes[b],
                    topic_embeddings,
                    active_topics,
                    topic_sizes,
                )

                current_score = self.compute_vendi_q_score(K)

                # Update cumulative mapping
                for t in cumulative_mapping:
                    if cumulative_mapping[t] == b:
                        cumulative_mapping[t] = a

                iteration += 1
                pbar.update(1)

                if self.verbose and iteration % 10 == 0:
                    print(
                        f"Iter {iteration}: {K.shape[0]} topics, "
                        f"ΔVendi={best_delta:.6f}, "
                        f"VS_q={current_score:.4f}"
                    )

        if self.verbose:
            print(f"Final: {len(active_topics)} topics after {iteration} merges")

        return cumulative_mapping
