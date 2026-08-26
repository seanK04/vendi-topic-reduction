"""
Vendi Score-based Topic Reduction using algebraic acceleration.

This module implements the Vendi₂ diversity score for iterative topic merging,
as described in the paper "Vendi Clustering for Topic Modeling".
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from tqdm import tqdm


class VendiReducer:
    """Implements Vendi₂-based topic reduction with lookahead algebraic acceleration."""
    
    def __init__(self, verbose: bool = False):
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
    
    def compute_all_deltas_vectorized(self, K: np.ndarray, T: float, R: np.ndarray, 
                                       G: np.ndarray, n_vec: np.ndarray):
        """
        Vectorized computation of Vendi delta for all pairs (a,b).
        Replaces the O(m^2) Python loop with O(m^2) BLAS operations.
        
        Args:
            K: Similarity matrix (m x m)
            T: Current trace value
            R: Row-wise squared sums (excluding diagonal)
            G: Gram matrix KK^T
            n_vec: Array of topic sizes
            
        Returns:
            best_delta: Maximum delta value
            best_T_new: Corresponding new T value
            idx_a: First topic index
            idx_b: Second topic index
        """
        m = K.shape[0]
        
        # Precompute combined weights and normalization constants
        n_sq = n_vec ** 2
        n_outer = np.outer(n_vec, n_vec)
        c2 = n_sq[:, None] + n_sq[None, :] + 2 * n_outer * K
        
        # Ensure positivity (avoid division by zero/negative)
        c2 = np.maximum(c2, 1e-10)
        
        # Compute sum_u_sq term efficiently using broadcasting
        R_col = R[:, None]
        R_row = R[None, :]
        
        Ra_prime = R_col - K**2
        Rb_prime = R_row - K**2
        Gab_prime = G - 2*K
        
        numerator = (n_sq[:, None] * Ra_prime) + \
                    (n_sq[None, :] * Rb_prime) + \
                    (2 * n_outer * Gab_prime)
        
        sum_u_sq = numerator / c2
        
        # Compute T_new for all pairs
        T_new_matrix = T - 2*R_col - 2*R_row + 2*(K**2) + 2*sum_u_sq - 1
        
        # Compute deltas
        vendi_old = (m ** 2) / T
        vendi_new_matrix = ((m - 1) ** 2) / T_new_matrix
        deltas = vendi_new_matrix - vendi_old
        
        # Mask diagonal and lower triangle to avoid duplicates/self-merges
        mask = np.triu(np.ones((m, m), dtype=bool), k=1)
        deltas_upper = np.where(mask, deltas, -np.inf)
        
        # Find best pair
        idx_flat = np.argmax(deltas_upper)
        idx_a, idx_b = np.unravel_index(idx_flat, deltas.shape)
        
        return deltas[idx_a, idx_b], T_new_matrix[idx_a, idx_b], idx_a, idx_b
    
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
        topic_embeddings: np.ndarray,
        active_topics: list,
        topic_sizes: dict
    ):
        m = K.shape[0]
        K_ab = K[idx_a, idx_b]
        c = np.sqrt(n_a**2 + n_b**2 + 2 * n_a * n_b * K_ab)
        
        k_u = (n_a * K[idx_a] + n_b * K[idx_b]) / c
        k_u[idx_a] = 1.0
        
        mask = np.ones(m, dtype=bool)
        mask[idx_b] = False
        
        K_new = K[np.ix_(mask, mask)]
        K_new[idx_a, :] = k_u[mask]
        K_new[:, idx_a] = k_u[mask]
        
        topic_embeddings[idx_a] = (
            n_a * topic_embeddings[idx_a]
            + n_b * topic_embeddings[idx_b]
        ) / (n_a + n_b)
        topic_sizes[active_topics[idx_a]] = n_a + n_b
        
        T_new = np.sum(K_new ** 2)
        R_new = np.sum(K_new ** 2, axis=1) - np.diag(K_new) ** 2
        
        k_a = K[:, idx_a]
        k_b = K[:, idx_b]
        G_new = G - np.outer(k_a, k_a) - np.outer(k_b, k_b) + np.outer(k_u, k_u)
        G_new = G_new[np.ix_(mask, mask)]
        
        topic_embeddings_new = np.delete(topic_embeddings, idx_b, axis=0)
        active_topics_new = [t for i, t in enumerate(active_topics) if i != idx_b]
        
        return K_new, T_new, R_new, G_new, active_topics_new, topic_embeddings_new
    
    def reduce(
        self,
        embeddings: np.ndarray,
        topic_sizes: Dict[int, int],
        target_k: Optional[int] = None
    ) -> Dict[int, int]:
        # Initialize active topics
        active_topics = sorted(topic_sizes.keys())
        m = len(active_topics)
        
        topic_embeddings = np.array([embeddings[t] for t in active_topics])
        K = cosine_similarity(topic_embeddings)

        # Initialize caches
        T, R, G = self.initialize_cache(K)
        vendi_score = self.compute_vendi2_score(T, m)

        if self.verbose:
            print(f"Initial: {m} topics, Vendi₂={vendi_score:.4f}")

        cumulative_mapping = {t: t for t in active_topics}

        iteration = 0
        total_merges = m - target_k if target_k is not None else m - 1
        
        with tqdm(total=total_merges, desc="    Vendi reduction", ncols=80,
                  bar_format='{desc}: {percentage:3.0f}%|{bar}| {n}/{total} [{elapsed}<{remaining}]',
                  disable=not target_k) as pbar:
            while True:
                m = K.shape[0]

                if target_k is not None and m <= target_k:
                    break
                if m <= 1:
                    break

                # Vectorized computation of all deltas
                n_vec = np.array([topic_sizes[t] for t in active_topics])
                best_delta, best_T_new, idx_a, idx_b = self.compute_all_deltas_vectorized(
                    K, T, R, G, n_vec
                )

                a = active_topics[idx_a]
                b = active_topics[idx_b]

                K, T, R, G, active_topics, topic_embeddings = self.apply_merge(
                    K, T, R, G,
                    idx_a, idx_b,
                    topic_sizes[a], topic_sizes[b],
                    topic_embeddings,
                    active_topics,
                    topic_sizes
                )

                # Update cumulative mapping
                for t in cumulative_mapping:
                    if cumulative_mapping[t] == b:
                        cumulative_mapping[t] = a

                iteration += 1
                pbar.update(1)
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