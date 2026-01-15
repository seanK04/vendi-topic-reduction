import numpy as np
from typing import Dict, List, Union
from sklearn.metrics.pairwise import cosine_similarity
from bertopic import BERTopic
from vendi_score import vendi

def compute_coherence_cv(topic_model: BERTopic, docs: List[str]) -> float:
    """
    Compute C_v coherence using Gensim with BERTopic-compatible tokenization.
    """
    try:
        from gensim.corpora import Dictionary
        from gensim.models import CoherenceModel
        
        # Use BERTopic's vectorizer analyzer for tokenization
        if hasattr(topic_model, 'vectorizer_model') and topic_model.vectorizer_model:
            analyzer = topic_model.vectorizer_model.build_analyzer()
            tokenized_docs = [analyzer(doc) for doc in docs]
        else:
            # Fallback to simple tokenization
            tokenized_docs = [doc.lower().split() for doc in docs]
        
        # Build dictionary from tokenized docs
        dictionary = Dictionary(tokenized_docs)
        
        # Extract topics with validation
        topics = []
        for topic_id in sorted(topic_model.get_topics().keys()):
            if topic_id == -1:  # Skip outlier topic
                continue
            
            # Get topic words and filter by dictionary
            topic_words = [word for word, _ in topic_model.get_topic(topic_id)[:10]]
            # Only keep words that exist in the dictionary
            topic_words = [w for w in topic_words if w in dictionary.token2id]
            
            # Require at least 2 words for meaningful coherence
            if len(topic_words) >= 2:
                topics.append(topic_words)
        
        if len(topics) == 0:
            return float('nan')  # Return NaN instead of 0 for failed calculation
        
        # Compute C_v coherence
        coherence_model = CoherenceModel(
            topics=topics,
            texts=tokenized_docs,
            dictionary=dictionary,
            coherence='c_v'
        )
        
        return coherence_model.get_coherence()
    
    except ImportError:
        print("Warning: gensim not installed. Install with: pip install gensim")
        return float('nan')
    except Exception as e:
        print(f"Warning: Could not compute C_v coherence: {e}")
        return float('nan')

def compute_coherence_npmi(topic_model: BERTopic, docs: List[str]) -> float:
    """
    Compute C_NPMI using Gensim with BERTopic-compatible tokenization.
    """
    try:
        from gensim.corpora import Dictionary
        from gensim.models import CoherenceModel
        
        # Use BERTopic's vectorizer analyzer for tokenization (fixes tokenization mismatch)
        if hasattr(topic_model, 'vectorizer_model') and topic_model.vectorizer_model:
            analyzer = topic_model.vectorizer_model.build_analyzer()
            tokenized_docs = [analyzer(doc) for doc in docs]
        else:
            # Fallback to simple tokenization
            tokenized_docs = [doc.lower().split() for doc in docs]
        
        # Build dictionary from tokenized docs
        dictionary = Dictionary(tokenized_docs)
        
        # Extract topics with validation
        topics = []
        for topic_id in sorted(topic_model.get_topics().keys()):
            if topic_id == -1:  # Skip outlier topic
                continue
            
            # Get topic words and filter by dictionary
            topic_words = [word for word, _ in topic_model.get_topic(topic_id)[:10]]
            # Only keep words that exist in the dictionary
            topic_words = [w for w in topic_words if w in dictionary.token2id]
            
            # Require at least 2 words for meaningful coherence
            if len(topic_words) >= 2:
                topics.append(topic_words)
        
        if len(topics) == 0:
            return float('nan')  # Return NaN instead of 0 for failed calculation
        
        # Compute C_NPMI coherence
        coherence_model = CoherenceModel(
            topics=topics,
            texts=tokenized_docs,
            dictionary=dictionary,
            coherence='c_npmi'
        )
        
        return coherence_model.get_coherence()
    
    except ImportError:
        print("Warning: gensim not installed. Install with: pip install gensim")
        return float('nan')
    except Exception as e:
        print(f"Warning: Could not compute C_NPMI coherence: {e}")
        return float('nan')

def compute_word_uniqueness(topic_model: BERTopic, top_n: int = 10) -> float:
    """
    Compute word uniqueness: proportion of unique words across all topics.
    """
    topics = topic_model.get_topics()
    
    all_words = []
    for topic_id, words in topics.items():
        if topic_id != -1:  # Skip outlier topic
            top_words = [word for word, _ in words[:top_n]]
            all_words.extend(top_words)
    
    if len(all_words) == 0:
        return 0.0
    
    unique_words = len(set(all_words))
    total_words = len(all_words)
    
    return unique_words / total_words
    

def compute_inter_topic_distance(topic_model: BERTopic) -> float:
    if topic_model.topic_embeddings_ is None:
        return 0.0
    
    # Exclude outlier topic if present
    if -1 in topic_model.get_topics():
        embeddings = topic_model.topic_embeddings_[1:]  # Skip first (outlier)
    else:
        embeddings = topic_model.topic_embeddings_
    
    if len(embeddings) < 2:
        return 0.0
    
    # Compute pairwise cosine similarity
    similarity_matrix = cosine_similarity(embeddings)
    
    # Extract upper triangle (excluding diagonal)
    n = similarity_matrix.shape[0]
    similarities = []
    for i in range(n):
        for j in range(i + 1, n):
            similarities.append(similarity_matrix[i, j])
    
    return 1.0 - np.mean(similarities) if similarities else 0.0

def compute_vendi_diversity(topic_model: BERTopic, q: Union[float, str] = 1.0) -> float:
    """
    Compute Vendi Diversity measuring between-topic distinctness using native topic embeddings.
    
    Uses BERTopic's pre-computed topic embeddings (centroids of document embeddings) 
    to measure the effective number of distinct topics via the Vendi Score.
    
    Args:
        topic_model: Fitted BERTopic model
        q: Q-value parameter for Vendi score (default: 1.0 for entropy-based)
    
    Returns:
        Vendi diversity score (higher = more diverse topics)
    """
    try:
        if topic_model.topic_embeddings_ is None:
            return 0.0
        
        # Exclude outlier topic if present
        if -1 in topic_model.get_topics():
            embeddings = topic_model.topic_embeddings_[1:]  # Skip first (outlier)
        else:
            embeddings = topic_model.topic_embeddings_
        
        if len(embeddings) < 2:
            return 0.0
        
        # Compute cosine similarity matrix between topic embeddings
        similarity_matrix = cosine_similarity(embeddings)
        
        # Compute Vendi Score on the similarity matrix (no normalization)
        vendi_score = vendi.score_K(similarity_matrix, q=q)
        
        return vendi_score
        
    except Exception as e:
        print(f"Warning: Could not compute Vendi diversity: {e}")
        return 0.0

def evaluate_model(
        topic_model: BERTopic,
        docs: List[str],
) -> Dict[str, float]:
    """
    Compute all evaluation metrics for a topic model.
    
    Args:
        topic_model: Fitted BERTopic model
        docs: List of documents
    
    Returns:
        Dictionary of metric names and values
    """
    metrics = {}
    topics = topic_model.topics_
    unique_topics = set(topics) - {-1} # Exclude outliers
    metrics["n_topics"] = len(unique_topics)
    metrics["n_outliers"] = sum(1 for t in topics if t == -1)
    metrics["outlier_ratio"] = metrics["n_outliers"] / len(topics)
    
    # Coherence Metrics
    metrics["coherence_cv"] = compute_coherence_cv(topic_model, docs)
    metrics["coherence_npmi"] = compute_coherence_npmi(topic_model, docs)

    # Diversity Metrics
    metrics["word_uniqueness_10"] = compute_word_uniqueness(topic_model, top_n=10)
    metrics["word_uniqueness_25"] = compute_word_uniqueness(topic_model, top_n=25)
    metrics["mean_intertopic_cosine"] = compute_inter_topic_distance(topic_model)

    # Vendi Metrics
    metrics["vendi_diversity_0.5"] = compute_vendi_diversity(topic_model, q=0.5)
    metrics["vendi_diversity_1"] = compute_vendi_diversity(topic_model, q=1.0)
    metrics["vendi_diversity_2"] = compute_vendi_diversity(topic_model, q=2.0)
    metrics["vendi_diversity_10"] = compute_vendi_diversity(topic_model, q=10.0)
    metrics["vendi_diversity_inf"] = compute_vendi_diversity(topic_model, q='inf')

    return metrics

def print_metrics(metrics:Dict[str, float], title: str = "Model Evaluation Results"):
    print(f"\n{'='*50}")
    print(f"{title:^50}")
    print(f"{'='*50}")

    # 1. Structural Stats
    print(f"\n[Structural Statistics]")
    print(f"  Number of Topics (k): {metrics.get('n_topics', 0)}")
    print(f"  Outlier Ratio:        {metrics.get('outlier_ratio', 0.0):.2%}")

    # 2. Quality / Coherence
    print(f"\n[Semantic Coherence]")
    npmi = metrics.get("coherence_npmi") or metrics.get("coherence_npmi", 0.0)
    print(f"  NPMI Coherence:       {npmi:.4f}")
    print(f"  C_v Coherence:        {metrics.get('coherence_cv', 0.0):.4f}")

    # 3. Diversity
    print(f"\n[Topic Diversity]")
    print(f"  Word Uniqueness @10:  {metrics.get('word_uniqueness_10', 0.0):.4f}")
    print(f"  Word Uniqueness @25:  {metrics.get('word_uniqueness_25', 0.0):.4f}")
    print(f"  Mean Inter-Topic Cos: {metrics.get('mean_intertopic_cosine', 0.0):.4f}")

    # 4. Vendi
    print(f"  Vendi Diversity 0.5:      {metrics.get('vendi_diversity_0.5', 0.0):.4f}")
    print(f"  Vendi Diversity 1:      {metrics.get('vendi_diversity_1', 0.0):.4f}")
    print(f"  Vendi Diversity 2:      {metrics.get('vendi_diversity_2', 0.0):.4f}")
    print(f"  Vendi Diversity 10:      {metrics.get('vendi_diversity_10', 0.0):.4f}")
    print(f"  Vendi Diversity inf:      {metrics.get('vendi_diversity_inf', 0.0):.4f}")
    
    print(f"\n{'='*50}\n")
