from dataclasses import dataclass
from typing import Any, Optional, TypeVar
from datetime import datetime
import re
import tiktoken
from openai import OpenAI
from loguru import logger
import json
RawPaperItem = TypeVar('RawPaperItem')

@dataclass
class RelatedPaper:
    title: str
    score: float
    abstract: Optional[str] = None


@dataclass
class ThemeReview:
    theme_score: float
    keep: bool
    reason: str
    matched_topic_id: Optional[str] = None
    lane: str = "core"
    object_match: Optional[float] = None
    method_match: Optional[float] = None
    question_match: Optional[float] = None
    context_match: Optional[float] = None
    novelty_score: Optional[float] = None
    boundary_violation: bool = False


@dataclass
class Paper:
    source: str
    title: str
    authors: list[str]
    abstract: str
    url: str
    pdf_url: Optional[str] = None
    full_text: Optional[str] = None
    tldr: Optional[str] = None
    affiliations: Optional[list[str]] = None
    score: Optional[float] = None
    related_papers: Optional[list[RelatedPaper]] = None
    theme_review: Optional[ThemeReview] = None
    topic_matches: Optional[list[Any]] = None
    matched_topic: Optional[Any] = None

    def _generate_tldr_with_llm(self, openai_client:OpenAI,llm_params:dict) -> str:
        lang = llm_params.get('language', 'English')
        prompt = f"Given the following information of a paper, generate a one-sentence TLDR summary in {lang}:\n\n"
        if self.title:
            prompt += f"Title:\n {self.title}\n\n"

        if self.abstract:
            prompt += f"Abstract: {self.abstract}\n\n"

        if self.full_text:
            prompt += f"Preview of main content:\n {self.full_text}\n\n"

        if not self.full_text and not self.abstract:
            logger.warning(f"Neither full text nor abstract is provided for {self.url}")
            return "Failed to generate TLDR. Neither full text nor abstract is provided"
        
        # use gpt-4o tokenizer for estimation
        enc = tiktoken.encoding_for_model("gpt-4o")
        prompt_tokens = enc.encode(prompt)
        prompt_tokens = prompt_tokens[:4000]  # truncate to 4000 tokens
        prompt = enc.decode(prompt_tokens)
        
        response = openai_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"You are an assistant who perfectly summarizes scientific paper, and gives the core idea of the paper to the user. Your answer should be in {lang}.",
                },
                {"role": "user", "content": prompt},
            ],
            **llm_params.get('generation_kwargs', {})
        )
        tldr = response.choices[0].message.content
        return tldr
    
    def generate_tldr(self, openai_client:OpenAI,llm_params:dict) -> str:
        try:
            tldr = self._generate_tldr_with_llm(openai_client,llm_params)
            self.tldr = tldr
            return tldr
        except Exception as e:
            logger.warning(f"Failed to generate tldr of {self.url}: {e}")
            tldr = self.abstract
            self.tldr = tldr
            return tldr

    def _generate_affiliations_with_llm(self, openai_client:OpenAI,llm_params:dict) -> Optional[list[str]]:
        if self.full_text is not None:
            prompt = f"Given the beginning of a paper, extract the affiliations of the authors in a python list format, which is sorted by the author order. If there is no affiliation found, return an empty list '[]':\n\n{self.full_text}"
            # use gpt-4o tokenizer for estimation
            enc = tiktoken.encoding_for_model("gpt-4o")
            prompt_tokens = enc.encode(prompt)
            prompt_tokens = prompt_tokens[:2000]  # truncate to 2000 tokens
            prompt = enc.decode(prompt_tokens)
            affiliations = openai_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant who perfectly extracts affiliations of authors from a paper. You should return a python list of affiliations sorted by the author order, like [\"TsingHua University\",\"Peking University\"]. If an affiliation is consisted of multi-level affiliations, like 'Department of Computer Science, TsingHua University', you should return the top-level affiliation 'TsingHua University' only. Do not contain duplicated affiliations. If there is no affiliation found, you should return an empty list [ ]. You should only return the final list of affiliations, and do not return any intermediate results.",
                    },
                    {"role": "user", "content": prompt},
                ],
                **llm_params.get('generation_kwargs', {})
            )
            affiliations = affiliations.choices[0].message.content

            affiliations = re.search(r'\[.*?\]', affiliations, flags=re.DOTALL).group(0)
            affiliations = json.loads(affiliations)
            affiliations = list(set(affiliations))
            affiliations = [str(a) for a in affiliations]

            return affiliations
    
    def generate_affiliations(self, openai_client:OpenAI,llm_params:dict) -> Optional[list[str]]:
        try:
            affiliations = self._generate_affiliations_with_llm(openai_client,llm_params)
            self.affiliations = affiliations
            return affiliations
        except Exception as e:
            logger.warning(f"Failed to generate affiliations of {self.url}: {e}")
            self.affiliations = None
            return None

    def _generate_theme_review_with_llm(self, openai_client:OpenAI,llm_params:dict) -> ThemeReview:
        lang = llm_params.get('language', 'English')
        matched_topic = getattr(self, "matched_topic", None)
        prompt = (
            "Judge whether the candidate paper is truly in the same research theme as the matched topic profile. "
            "Do not keep a paper just because it shares generic method words such as AI, model, optimization, embedding, "
            "Bayesian optimization, generative design, cancer, protein, or cell. The primary research object, task, "
            "biological context, and contribution must be aligned. For example, oncology drug-combination optimization "
            "and protein engineering should be considered different themes unless the candidate explicitly studies the "
            "same protein-engineering problem.\n\n"
            "Return only a JSON object with keys: matched_topic_id, theme_score, object_match, method_match, "
            "question_match, context_match, novelty_score, boundary_violation, decision, lane, reason. "
            "theme_score and match scores are 0 to 10. decision must be keep or drop. "
            "lane must be core, peripheral, or drop. reason should be one concise sentence "
            f"in {lang}.\n\n"
            f"Candidate title: {self.title}\n"
            f"Candidate abstract: {self.abstract}\n\n"
        )

        if matched_topic is not None:
            prompt += "Matched topic profile:\n"
            topic_id = getattr(matched_topic, "topic_id", "")
            topic_name = getattr(matched_topic, "topic_name", "")
            topic_score = getattr(matched_topic, "score", None)
            if topic_id:
                prompt += f"Topic ID: {topic_id}\n"
            if topic_name:
                prompt += f"Topic name: {topic_name}\n"
            if topic_score is not None:
                prompt += f"Topic embedding score: {float(topic_score):.1f}\n"
            profile = getattr(matched_topic, "profile", None)
            if profile is not None and hasattr(profile, "to_text"):
                prompt += f"{profile.to_text()}\n"
            prompt += "\nRepresentative Zotero papers:\n"
        else:
            prompt += "Matched topic profile: None recorded.\n\nClosest Zotero papers:\n"

        if self.related_papers:
            for i, related in enumerate(self.related_papers[:3], start=1):
                prompt += f"{i}. Title: {related.title}\n"
                prompt += f"   Similarity score: {related.score:.1f}\n"
                if related.abstract:
                    prompt += f"   Abstract: {related.abstract[:1600]}\n"
        else:
            prompt += "None recorded.\n"

        enc = tiktoken.encoding_for_model("gpt-4o")
        prompt_tokens = enc.encode(prompt)
        prompt_tokens = prompt_tokens[:5000]
        prompt = enc.decode(prompt_tokens)

        response = openai_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict research-topic relevance judge for an expert scientist.",
                },
                {"role": "user", "content": prompt},
            ],
            **llm_params.get('generation_kwargs', {})
        )
        content = response.choices[0].message.content or ""
        json_match = re.search(r'\{.*\}', content, flags=re.DOTALL)
        if json_match is None:
            raise ValueError(f"No JSON object found in theme review response: {content}")
        data = json.loads(json_match.group(0))
        theme_score = float(data.get("theme_score", 0))
        decision = str(data.get("decision", "drop")).strip().lower()
        matched_topic_id = data.get("matched_topic_id") or getattr(matched_topic, "topic_id", None)
        lane = str(data.get("lane", "core" if decision == "keep" else "drop")).strip().lower()
        reason = str(data.get("reason", "")).strip()
        return ThemeReview(
            theme_score=theme_score,
            keep=decision == "keep" and lane != "drop",
            reason=reason,
            matched_topic_id=str(matched_topic_id) if matched_topic_id else None,
            lane=lane,
            object_match=_optional_float(data.get("object_match")),
            method_match=_optional_float(data.get("method_match")),
            question_match=_optional_float(data.get("question_match")),
            context_match=_optional_float(data.get("context_match")),
            novelty_score=_optional_float(data.get("novelty_score")),
            boundary_violation=_as_bool(data.get("boundary_violation", False)),
        )

    def generate_theme_review(self, openai_client:OpenAI,llm_params:dict) -> Optional[ThemeReview]:
        try:
            self.theme_review = self._generate_theme_review_with_llm(openai_client,llm_params)
            return self.theme_review
        except Exception as e:
            logger.warning(f"Failed to generate theme review of {self.url}: {e}")
            self.theme_review = None
            return None
@dataclass
class CorpusPaper:
    title: str
    abstract: str
    added_date: datetime
    paths: list[str]


def _generate_daily_overview_with_llm(openai_client:OpenAI,llm_params:dict,papers:list[Paper]) -> str:
    lang = llm_params.get('language', 'English')
    prompt = (
        f"Write a concise daily research briefing in {lang} from the recommended papers below. "
        "Focus on helping the reader understand today's new research progress in their field. "
        "Include: 1) main themes, 2) notable methods or findings, and 3) papers worth reading first. "
        "Do not invent details beyond the provided titles, abstracts, TLDRs, and relevance evidence.\n\n"
    )
    for i, paper in enumerate(papers, start=1):
        prompt += f"Paper {i}\n"
        prompt += f"Source: {paper.source}\n"
        prompt += f"Title: {paper.title}\n"
        if paper.score is not None:
            prompt += f"Relevance score: {paper.score:.2f}\n"
        if paper.tldr:
            prompt += f"TLDR: {paper.tldr}\n"
        elif paper.abstract:
            prompt += f"Abstract: {paper.abstract}\n"
        if paper.related_papers:
            matches = "; ".join(f"{match.title} ({match.score:.1f})" for match in paper.related_papers[:3])
            prompt += f"Closest Zotero matches: {matches}\n"
        matched_topic = getattr(paper, "matched_topic", None)
        if matched_topic is not None:
            topic_id = getattr(matched_topic, "topic_id", "")
            topic_name = getattr(matched_topic, "topic_name", "")
            topic_score = getattr(matched_topic, "score", None)
            prompt += f"Matched topic: {topic_name} ({topic_id})"
            if topic_score is not None:
                prompt += f", score {float(topic_score):.1f}"
            prompt += "\n"
        if paper.theme_review:
            prompt += f"Theme score: {paper.theme_review.theme_score:.1f}\n"
            prompt += f"Theme lane: {paper.theme_review.lane}\n"
            detailed_scores = []
            for label, value in [
                ("object", paper.theme_review.object_match),
                ("method", paper.theme_review.method_match),
                ("question", paper.theme_review.question_match),
                ("context", paper.theme_review.context_match),
            ]:
                if value is not None:
                    detailed_scores.append(f"{label} {value:.1f}")
            if detailed_scores:
                prompt += f"Theme details: {', '.join(detailed_scores)}\n"
            prompt += f"Theme reason: {paper.theme_review.reason}\n"
        prompt += "\n"

    enc = tiktoken.encoding_for_model("gpt-4o")
    prompt_tokens = enc.encode(prompt)
    prompt_tokens = prompt_tokens[:6000]
    prompt = enc.decode(prompt_tokens)

    response = openai_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a research assistant who writes compact daily briefings for an expert researcher.",
            },
            {"role": "user", "content": prompt},
        ],
        **llm_params.get('generation_kwargs', {})
    )
    return response.choices[0].message.content


def generate_daily_overview(openai_client:OpenAI,llm_params:dict,papers:list[Paper]) -> Optional[str]:
    if len(papers) == 0:
        return None
    try:
        return _generate_daily_overview_with_llm(openai_client,llm_params,papers)
    except Exception as e:
        logger.warning(f"Failed to generate daily overview: {e}")
        return None


def _optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    return float(value)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)
