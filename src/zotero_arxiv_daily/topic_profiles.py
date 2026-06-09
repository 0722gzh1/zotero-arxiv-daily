"""Topic profile data model and configuration loading."""

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from loguru import logger
from omegaconf import DictConfig, ListConfig, OmegaConf


@dataclass
class TopicProfile:
    id: str
    name: str
    description: str
    core_objects: list[str]
    methods: list[str]
    research_questions: list[str]
    positive_examples: list[str]
    negative_boundaries: list[str]

    def to_text(self) -> str:
        """Return deterministic text suitable for embeddings or prompts."""
        lines = [
            f"Topic ID: {self.id}",
            f"Topic name: {self.name}",
            f"Description: {self.description}",
            "Core objects:",
            *_format_list(self.core_objects),
            "Methods:",
            *_format_list(self.methods),
            "Research questions:",
            *_format_list(self.research_questions),
            "Positive examples:",
            *_format_list(self.positive_examples),
            "Negative boundaries:",
            *_format_list(self.negative_boundaries),
        ]
        return "\n".join(lines)


def load_topic_profiles(config: Any) -> list[TopicProfile]:
    """Load enabled manually configured topic profiles."""
    topic_profiles_config = _get_nested(config, "topic_profiles")
    if topic_profiles_config is None:
        return []

    if not _get_nested(topic_profiles_config, "enabled"):
        return []

    topics = _to_plain_value(_get_nested(topic_profiles_config, "topics"))
    if not topics:
        topics = _load_topics_from_path(_get_nested(topic_profiles_config, "path"))
    if not topics:
        return []

    return [_topic_profile_from_config(topic) for topic in topics]


def generate_topic_profiles_from_corpus(
    corpus: list[Any],
    openai_client: Any,
    llm_params: Any,
    config: Any,
) -> list[TopicProfile]:
    """Generate temporary topic profiles from the current Zotero corpus."""
    if not corpus:
        return []

    topic_profiles_config = _get_nested(config, "topic_profiles")
    min_topics = int(_get_nested(topic_profiles_config, "min_topics") or 2)
    max_topics = int(_get_nested(topic_profiles_config, "max_topics") or 8)
    max_papers = int(_get_nested(topic_profiles_config, "max_papers_for_profile") or 80)
    topic_count = infer_topic_count(corpus, min_topics=min_topics, max_topics=max_topics)

    try:
        prompt = _build_auto_profile_prompt(corpus, topic_count, min_topics, max_topics, max_papers, llm_params)
        response = openai_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You generate stable research-interest topic profiles from a Zotero library.",
                },
                {"role": "user", "content": prompt},
            ],
            **_to_plain_value(_get_nested(llm_params, "generation_kwargs") or {}),
        )
        content = response.choices[0].message.content or ""
        topics = _parse_topics_json(content)
        profiles = [_topic_profile_from_config(topic) for topic in topics]
        if profiles:
            logger.info(f"Generated {len(profiles)} temporary topic profiles from Zotero corpus")
            return profiles
    except Exception as exc:
        logger.warning(f"Failed to generate topic profiles with LLM: {exc}")

    profiles = _fallback_profiles_from_collections(corpus, min_topics=min_topics, max_topics=max_topics)
    logger.info(f"Generated {len(profiles)} fallback topic profiles from Zotero collection paths")
    return profiles


def infer_topic_count(corpus: list[Any], min_topics: int = 2, max_topics: int = 8) -> int:
    """Infer a bounded topic count from Zotero library size and collection diversity."""
    if not corpus:
        return 0

    collection_roots = {
        _path_root(path)
        for paper in corpus
        for path in getattr(paper, "paths", []) or []
        if _path_root(path)
    }
    size_based = math.ceil(math.sqrt(len(corpus)) / 2.5)
    if collection_roots:
        diversity_based = math.ceil(len(collection_roots) / 2)
        count = max(size_based, diversity_based)
    else:
        count = size_based
    return max(min_topics, min(max_topics, count))


def _topic_profile_from_config(topic: Any) -> TopicProfile:
    topic_data = _to_mapping(topic)
    return TopicProfile(
        id=_to_string(topic_data.get("id", "")),
        name=_to_string(topic_data.get("name", "")),
        description=_to_string(topic_data.get("description", "")),
        core_objects=_to_string_list(topic_data.get("core_objects")),
        methods=_to_string_list(topic_data.get("methods")),
        research_questions=_to_string_list(topic_data.get("research_questions")),
        positive_examples=_to_string_list(topic_data.get("positive_examples")),
        negative_boundaries=_to_string_list(topic_data.get("negative_boundaries")),
    )


def _load_topics_from_path(path_value: Any) -> list[Any]:
    path = _to_plain_value(path_value)
    if not path:
        return []
    path = Path(str(path))
    if not path.exists():
        return []
    data = OmegaConf.load(path)
    return _to_plain_value(_get_nested(data, "topics")) or []


def _build_auto_profile_prompt(
    corpus: list[Any],
    topic_count: int,
    min_topics: int,
    max_topics: int,
    max_papers: int,
    llm_params: Any,
) -> str:
    lang = _get_nested(llm_params, "language") or "English"
    prompt = (
        f"Infer the number of topics from the Zotero library content, using about {topic_count} topics "
        f"bounded between {min_topics} and {max_topics}. "
        "Create temporary research-interest topic profiles for today's paper recommendation run. "
        "Topics should be stable research interests, not generic method words. "
        "Each topic must include explicit negative boundaries that prevent false positives.\n\n"
        "Return only JSON with this shape:\n"
        '{"topics":[{"id":"snake_case_id","name":"...","description":"...",'
        '"core_objects":["..."],"methods":["..."],"research_questions":["..."],'
        '"positive_examples":["Zotero paper title"],"negative_boundaries":["..."]}]}\n\n'
        f"Write names, descriptions, and boundaries in {lang}.\n\n"
    )
    for index, paper in enumerate(_select_profile_papers(corpus, max_papers), start=1):
        prompt += f"Zotero paper {index}\n"
        prompt += f"Title: {getattr(paper, 'title', '')}\n"
        paths = getattr(paper, "paths", []) or []
        if paths:
            prompt += f"Collections: {'; '.join(paths)}\n"
        abstract = getattr(paper, "abstract", "") or ""
        prompt += f"Abstract: {abstract[:1200]}\n\n"
    return prompt[:24000]


def _select_profile_papers(corpus: list[Any], max_papers: int) -> list[Any]:
    sorted_corpus = sorted(corpus, key=lambda paper: getattr(paper, "added_date", None), reverse=True)
    if len(sorted_corpus) <= max_papers:
        return sorted_corpus

    recent_count = max_papers // 2
    recent = sorted_corpus[:recent_count]
    remaining = sorted_corpus[recent_count:]
    step = max(1, len(remaining) // max(1, max_papers - recent_count))
    sampled = remaining[::step][: max_papers - recent_count]
    return recent + sampled


def _parse_topics_json(content: str) -> list[Any]:
    json_match = re.search(r'\{.*\}', content, flags=re.DOTALL)
    if json_match is None:
        raise ValueError(f"No JSON object found in topic profile response: {content}")
    data = json.loads(json_match.group(0), strict=False)
    topics = data.get("topics", [])
    if not isinstance(topics, list):
        raise ValueError("topic profile response field 'topics' must be a list")
    return topics


def _fallback_profiles_from_collections(corpus: list[Any], min_topics: int, max_topics: int) -> list[TopicProfile]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for paper in corpus:
        paths = getattr(paper, "paths", []) or []
        root = _path_root(paths[0]) if paths else "zotero_library"
        grouped[root].append(paper)

    groups = sorted(grouped.items(), key=lambda item: len(item[1]), reverse=True)[:max_topics]
    if len(groups) < min_topics and corpus:
        groups = [("zotero_library", corpus)]

    profiles = []
    for root, papers in groups:
        titles = [str(getattr(paper, "title", "")) for paper in papers if getattr(paper, "title", "")]
        name = _title_from_id(root)
        profiles.append(
            TopicProfile(
                id=_slug(root),
                name=name,
                description=f"Temporary topic inferred from Zotero collection '{name}'.",
                core_objects=_top_terms(papers, limit=5),
                methods=[],
                research_questions=[f"Research direction represented by Zotero collection '{name}'."],
                positive_examples=titles[:5],
                negative_boundaries=[
                    "generic method overlap without the same research object",
                    "papers matching only broad terms from this collection",
                ],
            )
        )
    return profiles


def _path_root(path: str) -> str:
    return str(path).split("/", 1)[0].strip()


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "zotero_library"


def _title_from_id(value: str) -> str:
    return re.sub(r"[_-]+", " ", value).strip().title() or "Zotero Library"


def _top_terms(papers: list[Any], limit: int) -> list[str]:
    stopwords = {
        "the", "and", "for", "with", "from", "into", "that", "this", "using", "paper",
        "study", "model", "models", "analysis", "data", "method", "methods", "new",
    }
    counts: dict[str, int] = defaultdict(int)
    for paper in papers:
        text = f"{getattr(paper, 'title', '')} {getattr(paper, 'abstract', '')}".lower()
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", text):
            if token not in stopwords:
                counts[token] += 1
    return [term for term, _ in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:limit]]


def _format_list(values: list[str]) -> list[str]:
    return [f"- {value}" for value in values]


def _get_nested(value: Any, path: str) -> Any:
    if value is None:
        return None

    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.select(value, path, default=None)

    current = value
    for part in path.split("."):
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(part)
        else:
            current = getattr(current, part, None)
    return current


def _to_mapping(value: Any) -> dict[str, Any]:
    plain_value = _to_plain_value(value)
    if isinstance(plain_value, dict):
        return plain_value
    if hasattr(plain_value, "__dict__"):
        return vars(plain_value)
    raise TypeError("topic profile entries must be mappings")


def _to_plain_value(value: Any) -> Any:
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _to_string(value: Any) -> str:
    return str(_to_plain_value(value))


def _to_string_list(value: Any) -> list[str]:
    plain_value = _to_plain_value(value)
    if plain_value is None:
        return []
    if isinstance(plain_value, str):
        return [plain_value]
    return [str(item) for item in plain_value]
