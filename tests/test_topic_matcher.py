from dataclasses import dataclass

import numpy as np
import pytest

from zotero_arxiv_daily.topic_matcher import TopicMatch, match_papers_to_topics


@dataclass
class Paperish:
    title: str
    abstract: str
    score: float | None = None
    related_papers: list | None = None


@dataclass
class TopicProfileish:
    id: str
    name: str
    body: str

    def to_text(self) -> str:
        return self.body


class StubReranker:
    def __init__(self, sim_matrix: np.ndarray):
        self.sim_matrix = sim_matrix
        self.candidate_texts = None
        self.topic_texts = None

    def get_similarity_score(self, candidate_texts, topic_texts):
        self.candidate_texts = candidate_texts
        self.topic_texts = topic_texts
        return self.sim_matrix


class ExplodingReranker:
    def get_similarity_score(self, candidate_texts, topic_texts):
        raise AssertionError("empty topic matching should not call the reranker")


def test_empty_topic_list_returns_same_papers_without_matching():
    papers = [
        Paperish(
            title="Candidate",
            abstract="A candidate abstract.",
        )
    ]

    result = match_papers_to_topics(papers, [], ExplodingReranker())

    assert result is papers
    assert not hasattr(papers[0], "topic_matches")
    assert not hasattr(papers[0], "matched_topic")


def test_two_papers_and_two_topics_get_sorted_matches_and_best_topic():
    papers = [
        Paperish(title="Protein design", abstract="Designing enzymes with language models."),
        Paperish(title="Drug combinations", abstract="Optimizing oncology drug combinations."),
    ]
    topics = [
        TopicProfileish("protein", "Protein Engineering", "Protein and enzyme design profile."),
        TopicProfileish("oncology", "Oncology Optimization", "Cancer therapy optimization profile."),
    ]
    reranker = StubReranker(
        np.array(
            [
                [0.2, 0.8],
                [0.9, 0.1],
            ]
        )
    )

    result = match_papers_to_topics(papers, topics, reranker)

    assert result is papers
    assert [paper.title for paper in result] == ["Protein design", "Drug combinations"]
    assert reranker.candidate_texts == [
        "Title: Protein design\nAbstract: Designing enzymes with language models.",
        "Title: Drug combinations\nAbstract: Optimizing oncology drug combinations.",
    ]
    assert reranker.topic_texts == [
        "Protein and enzyme design profile.",
        "Cancer therapy optimization profile.",
    ]

    first_matches = papers[0].topic_matches
    assert [match.topic_id for match in first_matches] == ["oncology", "protein"]
    assert [match.topic_name for match in first_matches] == ["Oncology Optimization", "Protein Engineering"]
    assert [match.score for match in first_matches] == pytest.approx([8.0, 2.0])
    assert all(isinstance(match, TopicMatch) for match in first_matches)
    assert first_matches[0].profile is topics[1]
    assert papers[0].matched_topic is first_matches[0]

    second_matches = papers[1].topic_matches
    assert [match.topic_id for match in second_matches] == ["protein", "oncology"]
    assert [match.score for match in second_matches] == pytest.approx([9.0, 1.0])
    assert second_matches[0].profile is topics[0]
    assert papers[1].matched_topic is second_matches[0]


def test_top_n_limits_topic_matches():
    papers = [Paperish(title="Candidate", abstract="A mixed research abstract.")]
    topics = [
        TopicProfileish("topic-a", "Topic A", "Profile A"),
        TopicProfileish("topic-b", "Topic B", "Profile B"),
        TopicProfileish("topic-c", "Topic C", "Profile C"),
    ]
    reranker = StubReranker(np.array([[0.3, 0.9, 0.6]]))

    match_papers_to_topics(papers, topics, reranker, top_n=2)

    assert [match.topic_id for match in papers[0].topic_matches] == ["topic-b", "topic-c"]
    assert [match.score for match in papers[0].topic_matches] == pytest.approx([9.0, 6.0])
    assert papers[0].matched_topic.topic_id == "topic-b"
