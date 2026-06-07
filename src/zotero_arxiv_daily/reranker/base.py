from abc import ABC, abstractmethod
from omegaconf import DictConfig
from ..protocol import Paper, CorpusPaper
import numpy as np
from typing import Type
class BaseReranker(ABC):
    def __init__(self, config:DictConfig):
        self.config = config

    def _get_relevance_setting(self, key: str, default):
        if self.config is None:
            return default
        reranker_config = self.config.get("reranker", {})
        relevance_config = reranker_config.get("relevance", {}) if reranker_config else {}
        return relevance_config.get(key, default) if relevance_config else default

    def rerank(self, candidates:list[Paper], corpus:list[CorpusPaper]) -> list[Paper]:
        corpus = sorted(corpus,key=lambda x: x.added_date,reverse=True)
        sim = self.get_similarity_score([c.abstract for c in candidates], [c.abstract for c in corpus])
        assert sim.shape == (len(candidates), len(corpus))
        top_k = max(1, int(self._get_relevance_setting("top_k", 20)))
        top_k = min(top_k, len(corpus))
        best_similarity_weight = float(self._get_relevance_setting("best_similarity_weight", 0.3))
        if not 0 <= best_similarity_weight <= 1:
            raise ValueError("reranker.relevance.best_similarity_weight must be between 0 and 1.")
        time_decay_strength = max(0.0, float(self._get_relevance_setting("time_decay_strength", 0.15)))

        if top_k == len(corpus):
            top_indices = np.tile(np.arange(len(corpus)), (len(candidates), 1))
        else:
            top_indices = np.argpartition(sim, -top_k, axis=1)[:, -top_k:]
        top_sim = np.take_along_axis(sim, top_indices, axis=1)

        time_decay_weight = 1 / (1 + np.log10(np.arange(len(corpus)) + 1))
        corpus_weights = 1 + time_decay_strength * time_decay_weight
        top_weights = corpus_weights[top_indices]
        top_weighted_sim = (top_sim * top_weights).sum(axis=1) / top_weights.sum(axis=1)
        best_sim = sim.max(axis=1)
        scores = ((1 - best_similarity_weight) * top_weighted_sim + best_similarity_weight * best_sim) * 10
        for s,c in zip(scores,candidates):
            c.score = s
        candidates = sorted(candidates,key=lambda x: x.score,reverse=True)
        return candidates
    
    @abstractmethod
    def get_similarity_score(self, s1:list[str], s2:list[str]) -> np.ndarray:
        raise NotImplementedError

registered_rerankers = {}

def register_reranker(name:str):
    def decorator(cls):
        registered_rerankers[name] = cls
        return cls
    return decorator

def get_reranker_cls(name:str) -> Type[BaseReranker]:
    if name not in registered_rerankers:
        raise ValueError(f"Reranker {name} not found")
    return registered_rerankers[name]
