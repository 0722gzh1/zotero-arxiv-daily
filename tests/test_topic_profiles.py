"""Tests for topic profile configuration loading and prompt text."""

from datetime import datetime
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest

from zotero_arxiv_daily.protocol import CorpusPaper


def test_load_topic_profiles_returns_empty_when_config_missing_or_disabled():
    from zotero_arxiv_daily.topic_profiles import load_topic_profiles

    assert load_topic_profiles(SimpleNamespace()) == []
    assert load_topic_profiles(
        SimpleNamespace(topic_profiles=SimpleNamespace(enabled=False))
    ) == []


def test_load_topic_profiles_returns_empty_when_topics_missing_or_empty():
    from zotero_arxiv_daily.topic_profiles import load_topic_profiles

    assert load_topic_profiles(
        SimpleNamespace(topic_profiles=SimpleNamespace(enabled=True))
    ) == []
    assert load_topic_profiles(
        SimpleNamespace(topic_profiles=SimpleNamespace(enabled=True, topics=[]))
    ) == []


def test_load_topic_profiles_from_omegaconf_converts_values_to_plain_python():
    from zotero_arxiv_daily.topic_profiles import load_topic_profiles

    config = OmegaConf.create(
        {
            "topic_profiles": {
                "enabled": True,
                "topics": [
                    {
                        "id": "robot-learning",
                        "name": "Robot Learning",
                        "description": "Learning policies for real-world robots.",
                        "core_objects": ["robot arms", "mobile manipulators"],
                        "methods": ["imitation learning", "offline RL"],
                        "research_questions": ["How can policies transfer safely?"],
                        "positive_examples": ["Diffusion policies for manipulation"],
                        "negative_boundaries": ["Pure simulation benchmarks"],
                    }
                ],
            }
        }
    )

    profiles = load_topic_profiles(config)

    assert len(profiles) == 1
    profile = profiles[0]
    assert profile.id == "robot-learning"
    assert profile.name == "Robot Learning"
    assert profile.description == "Learning policies for real-world robots."
    assert profile.core_objects == ["robot arms", "mobile manipulators"]
    assert profile.methods == ["imitation learning", "offline RL"]
    assert profile.research_questions == ["How can policies transfer safely?"]
    assert profile.positive_examples == ["Diffusion policies for manipulation"]
    assert profile.negative_boundaries == ["Pure simulation benchmarks"]
    assert type(profile.core_objects) is list
    assert type(profile.methods) is list
    assert type(profile.research_questions) is list
    assert type(profile.positive_examples) is list
    assert type(profile.negative_boundaries) is list


def test_topic_profile_to_text_includes_all_profile_fields():
    from zotero_arxiv_daily.topic_profiles import TopicProfile

    profile = TopicProfile(
        id="causal-rl",
        name="Causal RL",
        description="Causal methods for decision making.",
        core_objects=["causal graphs"],
        methods=["counterfactual evaluation"],
        research_questions=["Which interventions improve policies?"],
        positive_examples=["Causal representation learning"],
        negative_boundaries=["Generic supervised learning"],
    )

    assert profile.to_text() == "\n".join(
        [
            "Topic ID: causal-rl",
            "Topic name: Causal RL",
            "Description: Causal methods for decision making.",
            "Core objects:",
            "- causal graphs",
            "Methods:",
            "- counterfactual evaluation",
            "Research questions:",
            "- Which interventions improve policies?",
            "Positive examples:",
            "- Causal representation learning",
            "Negative boundaries:",
            "- Generic supervised learning",
        ]
    )


def test_infer_topic_count_grows_with_corpus_size_and_respects_limits():
    from zotero_arxiv_daily.topic_profiles import infer_topic_count

    small = [
        CorpusPaper("P1", "A1", datetime(2026, 1, 1), ["protein/design"]),
        CorpusPaper("P2", "A2", datetime(2026, 1, 2), ["protein/design"]),
    ]
    large = [
        CorpusPaper(f"P{i}", f"A{i}", datetime(2026, 1, 1), [f"topic-{i % 12}/sub"])
        for i in range(360)
    ]

    assert infer_topic_count(small, min_topics=2, max_topics=8) == 2
    assert infer_topic_count(large, min_topics=2, max_topics=8) == 8


def test_generate_topic_profiles_from_corpus_parses_llm_topics():
    from zotero_arxiv_daily.topic_profiles import generate_topic_profiles_from_corpus

    captured = {}

    def create(**kwargs):
        captured["messages"] = kwargs["messages"]
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            '{"topics": ['
                            '{"id": "protein_engineering", '
                            '"name": "Protein engineering", '
                            '"description": "Design proteins.", '
                            '"core_objects": ["protein sequence"], '
                            '"methods": ["generative models"], '
                            '"research_questions": ["optimize protein function"], '
                            '"positive_examples": ["Protein Paper"], '
                            '"negative_boundaries": ["generic oncology optimization"]}'
                            "]}",
                        ),
                    ),
                )
            ]
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    config = OmegaConf.create(
        {
            "topic_profiles": {
                "min_topics": 2,
                "max_topics": 6,
                "max_papers_for_profile": 10,
            }
        }
    )
    corpus = [
        CorpusPaper(
            "Protein Paper",
            "We design protein sequences.",
            datetime(2026, 1, 1),
            ["protein/design"],
        )
    ]

    profiles = generate_topic_profiles_from_corpus(
        corpus,
        client,
        {"language": "English", "generation_kwargs": {"model": "test-model"}},
        config,
    )

    assert len(profiles) == 1
    assert profiles[0].id == "protein_engineering"
    assert profiles[0].negative_boundaries == ["generic oncology optimization"]
    prompt = str(captured["messages"])
    assert "Infer the number of topics" in prompt
    assert "Zotero paper 1" in prompt
    assert "Protein Paper" in prompt


def test_generate_topic_profiles_from_corpus_falls_back_to_collection_paths_on_llm_error():
    from zotero_arxiv_daily.topic_profiles import generate_topic_profiles_from_corpus

    broken_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("api down"))
            )
        )
    )
    corpus = [
        CorpusPaper(
            "Protein A",
            "Protein design abstract.",
            datetime(2026, 1, 1),
            ["protein/design"],
        ),
        CorpusPaper(
            "Protein B",
            "Another protein abstract.",
            datetime(2026, 1, 2),
            ["protein/design"],
        ),
        CorpusPaper(
            "Oncology A",
            "Drug response abstract.",
            datetime(2026, 1, 3),
            ["oncology/drug-response"],
        ),
    ]

    profiles = generate_topic_profiles_from_corpus(
        corpus,
        broken_client,
        {"language": "English", "generation_kwargs": {"model": "test-model"}},
        OmegaConf.create({"topic_profiles": {"min_topics": 2, "max_topics": 5}}),
    )

    assert len(profiles) == 2
    assert {profile.id for profile in profiles} == {"protein", "oncology"}
    assert all(profile.positive_examples for profile in profiles)
