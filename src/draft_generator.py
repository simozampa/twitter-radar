"""
Draft content generator for trending topics.
Creates takes, threads, and article outlines based on detected trends.
"""

import json
import logging
from typing import Optional

logger = logging.getLogger("twitter_radar.drafts")


def build_trend_context(trend: dict) -> str:
    """Build a context string from a trend for draft generation."""
    lines = [
        f"TRENDING TOPIC: {trend['topic']}",
        f"Community: {trend['community']}",
        f"Velocity Score: {trend['velocity_score']:.1f}x (how fast it's accelerating)",
        f"Tweet Count: {trend['tweet_count']}",
        f"Keywords: {', '.join(trend['keywords'][:10])}",
        "",
        "TOP TWEETS (by engagement):",
    ]

    for i, tweet in enumerate(trend.get('top_tweets', [])[:5], 1):
        engagement = tweet.get('likes', 0) + tweet.get('retweets', 0) * 2
        lines.append(
            f"  {i}. @{tweet.get('author_username', '?')} "
            f"({tweet.get('likes', 0)} likes, {tweet.get('retweets', 0)} RTs): "
            f"{tweet['text'][:200]}"
        )

    return "\n".join(lines)


def generate_draft_prompt(trend: dict, style: dict) -> str:
    """
    Generate a prompt for draft creation.
    This returns a prompt string that can be used with any LLM.
    """
    context = build_trend_context(trend)
    style_name = style['name']
    style_desc = style['description']
    max_len = style.get('max_length', 1000)

    prompts = {
        "hot_take": f"""Based on this trending topic, write a spicy, opinionated hot take.
Be provocative but informed. Make it sound like a smart person with strong opinions, not a generic AI.
Keep it to 1-3 tweets max (280 chars each). Be direct, no hedging.

{context}

Write the hot take(s) now. No preamble, just the take.""",

        "thread": f"""Based on this trending topic, write an informative Twitter thread (5-8 tweets).
Break down what's happening, why it matters, and what to watch for.
Each tweet should be under 280 characters. Number them 1/, 2/, etc.
Make it insightful, not just a summary. Add analysis.

{context}

Write the thread now. No preamble.""",

        "article_outline": f"""Based on this trending topic, create a blog post / article outline.
Include:
- Compelling headline
- Key thesis (1-2 sentences)
- 4-6 main sections with bullet points
- Suggested data points or examples to include
- A contrarian angle or unique take

{context}

Write the outline now. No preamble.""",
    }

    return prompts.get(style_name, prompts["hot_take"])


def format_drafts_for_alert(drafts: list[dict]) -> str:
    """Format generated drafts for the Telegram alert message."""
    if not drafts:
        return ""

    lines = ["\n**Draft Takes:**"]
    for draft in drafts:
        style = draft.get('style', 'unknown')
        content = draft.get('content', '')
        emoji = {"hot_take": "🔥", "thread": "🧵", "article_outline": "📝"}.get(style, "📄")
        lines.append(f"\n{emoji} *{style.replace('_', ' ').title()}:*")
        lines.append(content)

    return "\n".join(lines)
