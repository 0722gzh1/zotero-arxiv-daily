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

        gen_kwargs = llm_params.get('generation_kwargs', {}) or {}
        response = openai_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"You are an assistant who perfectly summarizes scientific paper, and gives the core idea of the paper to the user. Your answer should be in {lang}.",
                },
                {"role": "user", "content": prompt},
            ],
            **gen_kwargs,
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
            prompt = f"Given the beginning of a paper, extract the affiliations of the authors in JSON list format, sorted by the author order. If no affiliation is found, return an empty JSON list []. Return only the JSON object with a single key 'affiliations'.\n\n{self.full_text}"
            # use gpt-4o tokenizer for estimation
            enc = tiktoken.encoding_for_model("gpt-4o")
            prompt_tokens = enc.encode(prompt)
            prompt_tokens = prompt_tokens[:2000]  # truncate to 2000 tokens
            prompt = enc.decode(prompt_tokens)
            data = _call_llm_for_json(
                openai_client,
                llm_params,
                prompt,
                system_prompt=(
                    "You extract author affiliations from a paper. Return a JSON object with a single key 'affiliations' "
                    "whose value is a JSON list of top-level institution names sorted by author order, deduplicated. "
                    "For multi-level affiliations keep only the top-level institution. If none, return {\"affiliations\": []}."
                ),
            )
            affiliations = data.get("affiliations", []) or []
            if not isinstance(affiliations, list):
                return None
            affiliations = list({str(a) for a in affiliations if a})
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
            "You judge whether a candidate paper would be useful to a research scientist working on the matched topic profile "
            "(or, if no profile is matched, on any of the scientist's closely related Zotero papers). "
            "Bias toward KEEP when the paper is in a clearly adjacent or useful direction; only DROP papers that are clearly "
            "outside the scientist's interests. The scientist cares about: protein engineering and design (including de novo "
            "design, protein language models, AlphaFold-guided engineering, directed evolution), nucleic acid design and "
            "engineering (RNA design, structure prediction, aptamers, therapeutic nucleic acids, nucleic acid dynamics tied "
            "to engineering), and suppressor tRNA / nonsense suppression / therapeutic readthrough (including orthogonal "
            "tRNA-synthetase pairs, in vivo PTC readthrough, translation fidelity, and tRNA delivery).\n\n"
            "Use the matched topic profile as the primary anchor, but you may also keep a paper if the closest Zotero matches "
            "below show a strong related signal even if the topic profile is a partial fit (e.g. cross-disciplinary tool, "
            "method transfer, or a therapeutic application of a tool the scientist works on). "
            "Do not drop a paper merely because it shares generic method words such as AI, model, optimization, embedding, "
            "Bayesian optimization, generative design, cancer, or protein. The primary research object, task, and contribution "
            "should be aligned with protein / nucleic acid design or suppressor tRNA work, OR the paper should be a clearly "
            "useful tool, mechanism, or therapeutic application that an expert in those areas would want to read.\n\n"
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

        data = self._call_llm_for_json(
            openai_client,
            llm_params,
            prompt,
            system_prompt="You are a research-topic relevance judge for an expert scientist. Bias toward KEEP for adjacent and useful work; only DROP papers that are clearly outside the scientist's interests.",
        )
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


def _extract_json_object(text: str) -> Optional[str]:
    """Extract the first balanced JSON object from a string.

    LLMs often wrap JSON in prose, code fences, or reasoning tokens. A naive
    re.search(r'\{.*\}', ..., flags=re.DOTALL) over-matches when there are
    multiple braces. This walks the string and returns the first balanced
    outermost object.
    """
    if not text:
        return None
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for end in range(start, len(text)):
            ch = text[end]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start : end + 1]
        start = text.find("{", start + 1)
    return None


def _repair_json_text(text: str) -> str:
    """Lightweight JSON repair for common LLM output mistakes.

    - Strips code fences
    - Replaces Python-style booleans/None
    - Removes trailing commas before } or ]
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = re.sub(r":\s*True\b", ":true", cleaned)
    cleaned = re.sub(r":\s*False\b", ":false", cleaned)
    cleaned = re.sub(r":\s*None\b", ":null", cleaned)
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    return cleaned


def _call_llm_for_json(
    openai_client: OpenAI,
    llm_params: dict,
    user_prompt: str,
    system_prompt: str,
    max_retries: int = 2,
) -> dict:
    """Call an OpenAI-compatible chat completion and parse a JSON object response.

    Retries up to ``max_retries`` times if the response is not valid JSON or
    cannot be extracted. The final attempt uses a stricter system prompt asking
    for raw JSON only.
    """
    gen_kwargs = llm_params.get("generation_kwargs", {}) or {}
    last_error: Optional[Exception] = None
    last_content: str = ""
    for attempt in range(max_retries + 1):
        system = system_prompt
        if attempt == max_retries:
            system = (
                "You must return only a single valid JSON object with no prose, "
                "no markdown, no code fences, and no commentary."
            )
        try:
            response = openai_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_prompt},
                ],
                **gen_kwargs,
            )
        except Exception as exc:
            last_error = exc
            logger.warning(f"LLM call failed on attempt {attempt + 1}: {exc}")
            continue

        content = (response.choices[0].message.content or "") if response.choices else ""
        last_content = content
        candidate = _extract_json_object(content)
        if candidate is None:
            last_error = ValueError(f"No JSON object found in LLM response: {content[:200]}")
            continue
        try:
            return json.loads(candidate, strict=False)
        except json.JSONDecodeError as exc:
            repaired = _repair_json_text(candidate)
            try:
                return json.loads(repaired, strict=False)
            except json.JSONDecodeError as exc2:
                last_error = exc2
                logger.warning(
                    f"LLM JSON parse failed on attempt {attempt + 1}: {exc2}; raw: {content[:200]}"
                )
                continue
    raise ValueError(
        f"LLM did not return valid JSON after {max_retries + 1} attempts; "
        f"last error: {last_error}; last content: {last_content[:200]}"
    )
