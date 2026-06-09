from dataclasses import dataclass


@dataclass
class TopicMatch:
    topic_id: str
    topic_name: str
    score: float
    profile: object


def _paper_embedding_text(paper: object) -> str:
    return f"Title: {paper.title}\nAbstract: {paper.abstract}"


def match_papers_to_topics(papers, topic_profiles, reranker, top_n=3):
    if not papers or not topic_profiles:
        return papers

    candidate_texts = [_paper_embedding_text(paper) for paper in papers]
    topic_texts = [profile.to_text() for profile in topic_profiles]
    similarity_scores = reranker.get_similarity_score(candidate_texts, topic_texts)
    match_count = min(top_n, len(topic_profiles))

    for paper_index, paper in enumerate(papers):
        matches = [
            TopicMatch(
                topic_id=str(topic_profiles[topic_index].id),
                topic_name=str(getattr(topic_profiles[topic_index], "name", topic_profiles[topic_index].id)),
                score=float(similarity_scores[paper_index, topic_index] * 10),
                profile=topic_profiles[topic_index],
            )
            for topic_index in range(len(topic_profiles))
        ]
        matches.sort(key=lambda match: match.score, reverse=True)
        paper.topic_matches = matches[:match_count]
        if paper.topic_matches:
            paper.matched_topic = paper.topic_matches[0]

    return papers
