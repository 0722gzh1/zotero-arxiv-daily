"""Tests for zotero_arxiv_daily.protocol: Paper LLM helpers."""

from types import SimpleNamespace

import pytest

from tests.canned_responses import make_sample_paper, make_stub_openai_client
from zotero_arxiv_daily.protocol import RelatedPaper


@pytest.fixture()
def llm_params():
    return {
        "language": "English",
        "generation_kwargs": {"model": "gpt-4o-mini", "max_tokens": 16384},
    }


# ---------------------------------------------------------------------------
# generate_tldr
# ---------------------------------------------------------------------------


def test_tldr_returns_response(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper()
    result = paper.generate_tldr(client, llm_params)
    assert result == "Hello! How can I assist you today?"
    assert paper.tldr == result


def test_tldr_without_abstract_or_fulltext(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper(abstract="", full_text=None)
    result = paper.generate_tldr(client, llm_params)
    assert "Failed to generate TLDR" in result


def test_tldr_falls_back_to_abstract_on_error(llm_params):
    paper = make_sample_paper()

    # Client whose create() raises
    from types import SimpleNamespace

    broken_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kw: (_ for _ in ()).throw(RuntimeError("API down")))
        )
    )
    result = paper.generate_tldr(broken_client, llm_params)
    assert result == paper.abstract


def test_tldr_truncates_long_prompt(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper(full_text="word " * 10000)
    result = paper.generate_tldr(client, llm_params)
    assert result is not None


# ---------------------------------------------------------------------------
# generate_affiliations
# ---------------------------------------------------------------------------


def test_affiliations_returns_parsed_list(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper()
    result = paper.generate_affiliations(client, llm_params)
    assert isinstance(result, list)
    assert "TsingHua University" in result
    assert "Peking University" in result


def test_affiliations_none_without_fulltext(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper(full_text=None)
    result = paper.generate_affiliations(client, llm_params)
    assert result is None


def test_affiliations_deduplicates(llm_params):
    """The stub returns two distinct affiliations, so no dedup needed.
    But confirm the set() dedup in the code doesn't break anything.
    """
    client = make_stub_openai_client()
    paper = make_sample_paper()
    result = paper.generate_affiliations(client, llm_params)
    assert len(result) == len(set(result))


def test_affiliations_malformed_llm_output(llm_params):
    """LLM returns affiliations without JSON brackets. Should fall back gracefully."""
    from types import SimpleNamespace

    def create_no_brackets(**kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="TsingHua University, Peking University"),
                )
            ]
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create_no_brackets)
        )
    )
    paper = make_sample_paper()
    result = paper.generate_affiliations(client, llm_params)
    # re.search for [...] will fail -> AttributeError -> caught -> returns None
    assert result is None


def test_affiliations_error_returns_none(llm_params):
    from types import SimpleNamespace

    broken_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")))
        )
    )
    paper = make_sample_paper()
    result = paper.generate_affiliations(broken_client, llm_params)
    assert result is None
    assert paper.affiliations is None


# ---------------------------------------------------------------------------
# generate_theme_review
# ---------------------------------------------------------------------------


def test_theme_review_parses_topic_profile_fields(llm_params):
    captured = {}

    def create_theme_review(**kwargs):
        captured["messages"] = kwargs["messages"]
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            '{"matched_topic_id": "protein_engineering", '
                            '"theme_score": 8.2, '
                            '"object_match": 9, '
                            '"method_match": 8, '
                            '"question_match": 8, '
                            '"context_match": 7, '
                            '"novelty_score": 6, '
                            '"boundary_violation": false, '
                            '"decision": "keep", '
                            '"lane": "core", '
                            '"reason": "Directly studies protein sequence optimization."}'
                        ),
                    ),
                )
            ]
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create_theme_review),
        )
    )
    topic_profile = SimpleNamespace(
        to_text=lambda: (
            "ID: protein_engineering\n"
            "Name: Protein engineering\n"
            "Negative boundaries: oncology drug-combination optimization"
        )
    )
    paper = make_sample_paper(
        title="Protein sequence optimization",
        abstract="We optimize protein sequences with a generative model.",
        related_papers=[
            RelatedPaper(
                title="Directed evolution in embedding space",
                score=8.5,
                abstract="A protein engineering study.",
            )
        ],
    )
    paper.matched_topic = SimpleNamespace(
        topic_id="protein_engineering",
        topic_name="Protein engineering",
        score=8.7,
        profile=topic_profile,
    )

    result = paper.generate_theme_review(client, llm_params)

    assert result.matched_topic_id == "protein_engineering"
    assert result.theme_score == pytest.approx(8.2)
    assert result.keep is True
    assert result.lane == "core"
    assert result.object_match == pytest.approx(9)
    assert result.method_match == pytest.approx(8)
    assert result.question_match == pytest.approx(8)
    assert result.context_match == pytest.approx(7)
    assert result.novelty_score == pytest.approx(6)
    assert result.boundary_violation is False
    assert result.reason == "Directly studies protein sequence optimization."

    prompt = str(captured["messages"])
    assert "Matched topic profile" in prompt
    assert "Negative boundaries: oncology drug-combination optimization" in prompt


def test_theme_review_defaults_topic_id_from_matched_topic(llm_params):
    def create_theme_review(**kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"theme_score": 7.1, "decision": "keep", "lane": "peripheral", "reason": "Useful method."}',
                    ),
                )
            ]
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create_theme_review),
        )
    )
    paper = make_sample_paper()
    paper.matched_topic = SimpleNamespace(
        topic_id="protein_engineering",
        topic_name="Protein engineering",
        score=7.2,
        profile=SimpleNamespace(to_text=lambda: "Protein engineering topic profile"),
    )

    result = paper.generate_theme_review(client, llm_params)

    assert result.matched_topic_id == "protein_engineering"
    assert result.lane == "peripheral"
    assert result.keep is True
