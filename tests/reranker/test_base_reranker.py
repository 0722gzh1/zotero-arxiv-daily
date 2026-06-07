"""Tests for BaseReranker: scoring, sorting, time decay, unknown reranker."""

import numpy as np
import pytest
from omegaconf import OmegaConf

from zotero_arxiv_daily.reranker.base import BaseReranker, get_reranker_cls
from tests.canned_responses import make_sample_paper, make_sample_corpus


class StubReranker(BaseReranker):
    """Reranker with a controlled similarity matrix for deterministic tests."""

    def __init__(self, sim_matrix: np.ndarray, config=None):
        self.config = config
        self._sim = sim_matrix
        self.s1 = None
        self.s2 = None

    def get_similarity_score(self, s1, s2):
        self.s1 = s1
        self.s2 = s2
        return self._sim


def test_rerank_scores_and_sorts():
    corpus = make_sample_corpus(3)
    papers = [make_sample_paper(title=f"Paper {i}") for i in range(2)]

    # Paper 1 has higher similarity to all corpus papers
    sim = np.array([
        [0.1, 0.1, 0.1],  # paper 0 — low
        [0.9, 0.9, 0.9],  # paper 1 — high
    ])
    reranker = StubReranker(sim)
    ranked = reranker.rerank(papers, corpus)
    assert ranked[0].title == "Paper 1"
    assert ranked[1].title == "Paper 0"
    assert ranked[0].score > ranked[1].score


def test_rerank_time_decay_weighting():
    corpus = make_sample_corpus(3)
    papers = [make_sample_paper(title="P")]

    # Only similar to the oldest paper (index 2 after reverse-sort by date)
    sim = np.array([[0.0, 0.0, 1.0]])
    reranker = StubReranker(sim)
    ranked_old = reranker.rerank(papers, corpus)
    score_old = ranked_old[0].score

    # Only similar to the newest paper (index 0 after reverse-sort by date)
    papers2 = [make_sample_paper(title="P")]
    sim2 = np.array([[1.0, 0.0, 0.0]])
    reranker2 = StubReranker(sim2)
    ranked_new = reranker2.rerank(papers2, corpus)
    score_new = ranked_new[0].score

    # Newest corpus paper gets higher time-decay weight, so score should be higher
    assert score_new > score_old


def test_rerank_prefers_strong_nearest_neighbors_over_broad_matches():
    corpus = make_sample_corpus(6)
    papers = [
        make_sample_paper(title="Broad Match"),
        make_sample_paper(title="Focused Match"),
    ]
    sim = np.array([
        [0.50, 0.50, 0.50, 0.50, 0.50, 0.50],
        [0.95, 0.20, 0.20, 0.20, 0.20, 0.20],
    ])
    config = OmegaConf.create({
        "reranker": {
            "relevance": {
                "top_k": 2,
                "best_similarity_weight": 0.3,
                "time_decay_strength": 0.0,
            }
        }
    })
    reranker = StubReranker(sim, config)
    ranked = reranker.rerank(papers, corpus)

    assert ranked[0].title == "Focused Match"
    assert ranked[0].score > ranked[1].score


def test_rerank_uses_title_and_abstract_for_embedding_input():
    corpus = make_sample_corpus(1)
    papers = [make_sample_paper(title="Candidate Title", abstract="Candidate Abstract")]
    sim = np.array([[0.5]])
    reranker = StubReranker(sim)

    reranker.rerank(papers, corpus)

    assert reranker.s1 == ["Title: Candidate Title\nAbstract: Candidate Abstract"]
    assert "Title: Corpus Paper 0" in reranker.s2[0]
    assert "Abstract: Abstract for corpus paper 0." in reranker.s2[0]


def test_rerank_records_closest_zotero_matches():
    corpus = make_sample_corpus(3)
    papers = [make_sample_paper(title="Candidate")]
    sim = np.array([[0.2, 0.9, 0.5]])
    config = OmegaConf.create({"reranker": {"relevance": {"match_count": 2}}})
    reranker = StubReranker(sim, config)

    ranked = reranker.rerank(papers, corpus)

    assert [match.title for match in ranked[0].related_papers] == ["Corpus Paper 1", "Corpus Paper 2"]
    assert ranked[0].related_papers[0].score == pytest.approx(9.0)


def test_rerank_single_candidate_single_corpus():
    corpus = make_sample_corpus(1)
    papers = [make_sample_paper()]
    sim = np.array([[0.5]])
    reranker = StubReranker(sim)
    ranked = reranker.rerank(papers, corpus)
    assert len(ranked) == 1
    assert ranked[0].score is not None


def test_get_reranker_cls_unknown():
    with pytest.raises(ValueError, match="not found"):
        get_reranker_cls("nonexistent_reranker_xyz")
