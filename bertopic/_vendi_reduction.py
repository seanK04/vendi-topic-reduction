"""
Vendi Score-based Topic Reduction using algebraic acceleration.

This module implements the Vendi₂ diversity score for iterative topic merging,
as described in the paper "Vendi Clustering for Topic Modeling".
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict


class VendiReducer:
    """Implements Vendi₂-based topic reduction with lookahead algebraic acceleration."""
    
    def __init__(self, epsilon: float = 1e-5, verbose: bool = False):
        self.epsilon = epsilon
        self.verbose = verbose
    
    def compute_vendi2_score(self, T: float, m: int) -> float:
        return (m ** 2) / T if T > 0 else 0.0
    
    def initialize_cache(self, K: np.ndarray):
        """
        Initialize cached quantities:
        T  = ||K||_F^2
        R  = row-wise squared sums (excluding diagonal)
        G  = Gram matrix KK^T
        """
        T = np.sum(K ** 2)
        R = np.sum(K ** 2, axis=1) - np.diag(K) ** 2
        G = K @ K.T
        return T, R, G
    
    def compute_merge_delta(
        self,
        K: np.ndarray,
        T: float,
        R: np.ndarray,
        G: np.ndarray,
        idx_a: int,
        idx_b: int,
        n_a: float,
        n_b: float
    ):
        """
        O(1) merge scoring using lookahead cache.
        """
        m = K.shape[0]
        K_ab = K[idx_a, idx_b]
        
        c2 = n_a**2 + n_b**2 + 2 * n_a * n_b * K_ab
        if c2 <= 0:
            return -np.inf, None
        
        # Σ_{j≠a,b} K_uj^2 via cached quantities
        Ra = R[idx_a] - K_ab**2
        Rb = R[idx_b] - K_ab**2
        Gab = G[idx_a, idx_b] - 2 * K_ab
        
        sum_u_sq = (
            n_a**2 * Ra
            + n_b**2 * Rb
            + 2 * n_a * n_b * Gab
        ) / c2
        
        # Exact Frobenius update
        T_new = (
            T
            - 2 * R[idx_a]
            - 2 * R[idx_b]
            + 2 * K_ab**2
            + 2 * sum_u_sq
            - 1
        )
        
        vendi_old = self.compute_vendi2_score(T, m)
        vendi_new = self.compute_vendi2_score(T_new, m - 1)
        
        return vendi_new - vendi_old, T_new
    
    def apply_merge(
        self,
        K: np.ndarray,
        T: float,
        R: np.ndarray,
        G: np.ndarray,
        idx_a: int,
        idx_b: int,
        n_a: float,
        n_b: float,
        embeddings: np.ndarray,
        active_topics: list,
        topic_sizes: dict
    ):
        """
        Apply merge a <- b and update K, R, G exactly.
        """
        m = K.shape[0]
        K_ab = K[idx_a, idx_b]
        c = np.sqrt(n_a**2 + n_b**2 + 2 * n_a * n_b * K_ab)
        
        # Build merged similarity vector k_u
        k_u = (n_a * K[idx_a] + n_b * K[idx_b]) / c
        k_u[idx_a] = 1.0
        
        # Remove b
        mask = np.ones(m, dtype=bool)
        mask[idx_b] = False
        
        K_new = K[np.ix_(mask, mask)]
        K_new[idx_a, :] = k_u[mask]
        K_new[:, idx_a] = k_u[mask]
        
        # Update embeddings and sizes
        embeddings[active_topics[idx_a]] = (
            n_a * embeddings[active_topics[idx_a]]
            + n_b * embeddings[active_topics[idx_b]]
        ) / (n_a + n_b)
        topic_sizes[active_topics[idx_a]] = n_a + n_b
        
        # Update caches
        T_new = np.sum(K_new ** 2)
        R_new = np.sum(K_new ** 2, axis=1) - np.diag(K_new) ** 2
        
        # Update G via rank-1 update
        k_a = K[:, idx_a]
        k_b = K[:, idx_b]
        G_new = G - np.outer(k_a, k_a) - np.outer(k_b, k_b) + np.outer(k_u, k_u)
        G_new = G_new[np.ix_(mask, mask)]
        
        # Update topic list
        active_topics_new = [t for i, t in enumerate(active_topics) if i != idx_b]
        
        return K_new, T_new, R_new, G_new, active_topics_new
    
    def reduce(
        self,
        embeddings: np.ndarray,
        topic_sizes: Dict[int, int],
        target_k: Optional[int] = None,
        epsilon: Optional[float] = None
    ) -> Dict[int, int]:

        if epsilon is None:
            epsilon = self.epsilon

        # Initialize active topics
        active_topics = list(topic_sizes.keys())
        m = len(active_topics)

        # Compute initial similarity matrix
        K = cosine_similarity(embeddings[active_topics])

        # Initialize caches
        T, R, G = self.initialize_cache(K)
        vendi_score = self.compute_vendi2_score(T, m)

        if self.verbose:
            print(f"Initial: {m} topics, Vendi₂={vendi_score:.4f}")

        # Track cumulative merges
        cumulative_mapping = {t: t for t in active_topics}

        iteration = 0
        while True:
            m = K.shape[0]

            # Stopping conditions
            if target_k is not None and m <= target_k:
                break
            if m <= 1:
                break

            best_delta = -np.inf
            best_pair = None
            best_T_new = None

            # === O(m^2) exhaustive search ===
            for idx_a in range(m):
                a = active_topics[idx_a]
                n_a = topic_sizes[a]

                for idx_b in range(idx_a + 1, m):
                    b = active_topics[idx_b]
                    n_b = topic_sizes[b]

                    delta, T_new = self.compute_merge_delta(
                        K, T, R, G,
                        idx_a, idx_b,
                        n_a, n_b
                    )

                    if delta > best_delta:
                        best_delta = delta
                        best_pair = (idx_a, idx_b)
                        best_T_new = T_new

            # Epsilon stopping
            if target_k is None and best_delta < -epsilon:
                if self.verbose:
                    print(f"Stopping: ΔVendi={best_delta:.6f} < -ε")
                break

            idx_a, idx_b = best_pair
            a = active_topics[idx_a]
            b = active_topics[idx_b]

            # === Apply merge ===
            K, T, R, G, active_topics = self.apply_merge(
                K, T, R, G,
                idx_a, idx_b,
                topic_sizes[a], topic_sizes[b],
                embeddings,
                active_topics,
                topic_sizes
            )

            # Update cumulative mapping
            for t in cumulative_mapping:
                if cumulative_mapping[t] == b:
                    cumulative_mapping[t] = a

            iteration += 1
            vendi_score = self.compute_vendi2_score(T, K.shape[0])

            if self.verbose and iteration % 10 == 0:
                print(
                    f"Iter {iteration}: {K.shape[0]} topics, "
                    f"ΔVendi={best_delta:.6f}, "
                    f"Vendi₂={vendi_score:.4f}"
                )

        if self.verbose:
            print(f"Final: {len(active_topics)} topics after {iteration} merges")

        return cumulative_mapping
