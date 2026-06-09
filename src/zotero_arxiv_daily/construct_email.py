from .protocol import Paper
import math
from html import escape


framework = """
<!DOCTYPE HTML>
<html>
<head>
  <style>
    .star-wrapper {
      font-size: 1.3em; /* 调整星星大小 */
      line-height: 1; /* 确保垂直对齐 */
      display: inline-flex;
      align-items: center; /* 保持对齐 */
    }
    .half-star {
      display: inline-block;
      width: 0.5em; /* 半颗星的宽度 */
      overflow: hidden;
      white-space: nowrap;
      vertical-align: middle;
    }
    .full-star {
      vertical-align: middle;
    }
  </style>
</head>
<body>

<div>
    __CONTENT__
</div>

<br><br>
<div>
To unsubscribe, remove your email in your Github Action setting.
</div>

</body>
</html>
"""

def get_empty_html():
  block_template = """
  <table border="0" cellpadding="0" cellspacing="0" width="100%" style="font-family: Arial, sans-serif; border: 1px solid #ddd; border-radius: 8px; padding: 16px; background-color: #f9f9f9;">
  <tr>
    <td style="font-size: 20px; font-weight: bold; color: #333;">
        No Papers Today. Take a Rest!
    </td>
  </tr>
  </table>
  """
  return block_template

def text_to_html(text:str | None) -> str:
    if not text:
        return ""
    return "<br>".join(escape(str(text)).splitlines())


def get_overview_html(daily_overview:str | None):
    if not daily_overview:
        return ""
    block_template = """
    <table border="0" cellpadding="0" cellspacing="0" width="100%" style="font-family: Arial, sans-serif; border: 1px solid #cfd8dc; border-radius: 8px; padding: 16px; background-color: #eef7f8;">
    <tr>
        <td style="font-size: 20px; font-weight: bold; color: #263238;">
            Daily Research Briefing
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #263238; padding: 10px 0 0 0; line-height: 1.5;">
            {daily_overview}
        </td>
    </tr>
</table>
"""
    return block_template.format(daily_overview=text_to_html(daily_overview))


def get_related_papers_html(paper:Paper) -> str:
    if not paper.related_papers:
        return ""
    items = []
    for related in paper.related_papers[:3]:
        items.append(f"{escape(related.title)} ({related.score:.1f})")
    return "; ".join(items)


def _format_score(value) -> str:
    try:
        return f"{float(value):.1f}/10"
    except (TypeError, ValueError):
        return text_to_html(value)


def get_matched_topic_html(paper:Paper) -> str:
    matched_topic = getattr(paper, "matched_topic", None)
    if matched_topic is None:
        return ""
    topic_id = getattr(matched_topic, "topic_id", None)
    topic_name = getattr(matched_topic, "topic_name", None)
    score = getattr(matched_topic, "score", None)
    if topic_id is None or topic_name is None or score is None:
        return ""
    return (
        f"<br><strong>Matched topic:</strong> {text_to_html(topic_name)} "
        f"({text_to_html(topic_id)}, {_format_score(score)})"
    )


def get_theme_review_html(paper:Paper) -> str:
    if paper.theme_review is None:
        return ""
    review = paper.theme_review
    decision = "keep" if review.keep else "drop"
    details = []
    lane = getattr(review, "lane", None)
    if lane is not None:
        details.append(f"<br><strong>Lane:</strong> {text_to_html(lane)}")
    for attr, label in (
        ("object_match", "Object match"),
        ("method_match", "Method match"),
        ("question_match", "Question match"),
        ("context_match", "Context match"),
    ):
        value = getattr(review, attr, None)
        if value is not None:
            details.append(f"<br><strong>{label}:</strong> {_format_score(value)}")
    boundary_violation = getattr(review, "boundary_violation", None)
    if boundary_violation is not None:
        boundary_text = "Yes" if boundary_violation else "No"
        details.append(f"<br><strong>Boundary violation:</strong> {boundary_text}")
    return (
        f"<br><strong>Theme match:</strong> {review.theme_score:.1f}/10 ({decision})"
        f"{''.join(details)}"
        f"<br><strong>Theme reason:</strong> {text_to_html(review.reason)}"
    )


def get_block_html(title:str, authors:str, rate:str, tldr:str, pdf_url:str, affiliations:str=None, source:str=None, related_papers:str=None, theme_review:str=None, matched_topic:str=None):
    block_template = """
    <table border="0" cellpadding="0" cellspacing="0" width="100%" style="font-family: Arial, sans-serif; border: 1px solid #ddd; border-radius: 8px; padding: 16px; background-color: #f9f9f9;">
    <tr>
        <td style="font-size: 20px; font-weight: bold; color: #333;">
            {title}
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #666; padding: 8px 0;">
            {authors}
            <br>
            <i>{affiliations}</i>
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>Source:</strong> {source}<br>
            <strong>Relevance:</strong> {rate}
            {matched_topic}
            <br>
            <strong>Recommended because:</strong> closest to Zotero papers: {related_papers}
            {theme_review}
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>TLDR:</strong> {tldr}
        </td>
    </tr>

    <tr>
        <td style="padding: 8px 0;">
            <a href="{pdf_url}" style="display: inline-block; text-decoration: none; font-size: 14px; font-weight: bold; color: #fff; background-color: #d9534f; padding: 8px 16px; border-radius: 4px;">PDF</a>
        </td>
    </tr>
</table>
"""
    return block_template.format(
        title=text_to_html(title),
        authors=text_to_html(authors),
        rate=text_to_html(rate),
        tldr=text_to_html(tldr),
        pdf_url=escape(str(pdf_url)),
        affiliations=text_to_html(affiliations),
        source=text_to_html(source or "Unknown"),
        related_papers=related_papers or "No close Zotero match recorded",
        theme_review=theme_review or "",
        matched_topic=matched_topic or "",
    )

def get_stars(score:float):
    full_star = '<span class="full-star">&#9733;</span>'
    half_star = '<span class="half-star">&#9733;</span>'
    low = 6
    high = 8
    if score <= low:
        return ''
    elif score >= high:
        return full_star * 5
    else:
        interval = (high-low) / 10
        star_num = math.ceil((score-low) / interval)
        full_star_num = int(star_num/2)
        half_star_num = star_num - full_star_num * 2
        return '<div class="star-wrapper">'+full_star * full_star_num + half_star * half_star_num + '</div>'


def render_email(papers:list[Paper], daily_overview:str | None=None) -> str:
    parts = []
    if len(papers) == 0 :
        return framework.replace('__CONTENT__', get_empty_html())

    overview_html = get_overview_html(daily_overview)
    if overview_html:
        parts.append(overview_html)
    
    for p in papers:
        #rate = get_stars(p.score)
        rate = round(p.score, 1) if p.score is not None else 'Unknown'
        author_list = [a for a in p.authors]
        num_authors = len(author_list)
        if num_authors <= 5:
            authors = ', '.join(author_list)
        else:
            authors = ', '.join(author_list[:3] + ['...'] + author_list[-2:])
        if p.affiliations is not None:
            affiliations = p.affiliations[:5]
            affiliations = ', '.join(affiliations)
            if len(p.affiliations) > 5:
                affiliations += ', ...'
        else:
            affiliations = 'Unknown Affiliation'
        related_papers = get_related_papers_html(p)
        theme_review = get_theme_review_html(p)
        matched_topic = get_matched_topic_html(p)
        parts.append(get_block_html(p.title, authors, rate, p.tldr, p.pdf_url, affiliations, p.source, related_papers, theme_review, matched_topic))

    content = '<br>' + '</br><br>'.join(parts) + '</br>'
    return framework.replace('__CONTENT__', content)
