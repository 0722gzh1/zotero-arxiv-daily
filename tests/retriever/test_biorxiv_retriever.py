"""Tests for BiorxivRetriever."""

from omegaconf import open_dict

from zotero_arxiv_daily.retriever.biorxiv_retriever import BiorxivRetriever
from tests.canned_responses import SAMPLE_BIORXIV_API_RESPONSE


def test_biorxiv_retrieve(config, mock_biorxiv_api, monkeypatch):
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)
    with open_dict(config.source):
        config.source.biorxiv = {"category": ["bioinformatics"]}
    retriever = BiorxivRetriever(config)
    papers = retriever.retrieve_papers()
    # Only latest date. bioRxiv categories are intentionally not filtered.
    assert len(papers) == 2
    assert [p.title for p in papers] == ["A biorxiv paper", "Another biorxiv paper"]


def test_biorxiv_empty_response(config, monkeypatch):
    import requests
    from types import SimpleNamespace

    empty = {"messages": [{"status": "ok"}], "collection": []}

    def _patched(url, **kw):
        resp = SimpleNamespace(status_code=200, raise_for_status=lambda: None)
        resp.json = lambda: empty
        return resp

    monkeypatch.setattr(requests, "get", _patched)
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)

    with open_dict(config.source):
        config.source.biorxiv = {"category": ["bioinformatics"]}
    retriever = BiorxivRetriever(config)
    papers = retriever.retrieve_papers()
    assert papers == []


def test_biorxiv_convert_to_paper(config):
    with open_dict(config.source):
        config.source.biorxiv = {"category": ["bioinformatics"]}
    retriever = BiorxivRetriever(config)
    raw = SAMPLE_BIORXIV_API_RESPONSE["collection"][0]
    paper = retriever.convert_to_paper(raw)
    assert paper.title == "A biorxiv paper"
    assert paper.source == "biorxiv"
    assert "biorxiv.org" in paper.pdf_url
    assert paper.authors == ["Smith, J.", "Doe, A.", "Lee, K."]


def test_biorxiv_does_not_require_category(config):
    with open_dict(config.source):
        config.source.biorxiv = {"category": None}
    retriever = BiorxivRetriever(config)
    assert retriever.retriever_config.category is None
