"""
Trend detection engine.
Identifies trending topics by analyzing engagement velocity,
keyword clustering, and conversation patterns.
"""

import re
import math
import time
import logging
from collections import Counter, defaultdict
from typing import Optional

logger = logging.getLogger("twitter_radar.detector")

# Common stop words to filter out
STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "don", "now", "and", "but", "or", "if", "this", "that", "these",
    "those", "it", "its", "i", "me", "my", "we", "our", "you", "your",
    "he", "him", "his", "she", "her", "they", "them", "their", "what",
    "which", "who", "whom", "about", "up", "like", "get", "got", "one",
    "also", "new", "even", "still", "back", "going", "much", "way",
    "really", "think", "know", "see", "come", "make", "right", "say",
    "said", "well", "good", "great", "time", "people", "want", "look",
    "https", "http", "amp", "rt", "via", "lol", "lmao", "gonna", "gotta",
    "thing", "things", "yeah", "yes", "hey", "let", "take", "every",
}


def extract_keywords(text: str) -> list[str]:
    """Extract meaningful keywords from tweet text."""
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    # Extract hashtags separately
    hashtags = re.findall(r'#(\w+)', text.lower())
    # Extract cashtags
    cashtags = re.findall(r'\$([A-Za-z]+)', text)
    # Clean and tokenize
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
    words = text.split()
    # Filter
    keywords = [w for w in words if len(w) > 2 and w not in STOP_WORDS]
    # Combine, prioritize hashtags and cashtags
    return cashtags + hashtags + keywords


def compute_engagement_score(tweet: dict) -> float:
    """
    Weighted engagement score.
    Quotes and retweets are higher signal than likes.
    Priority accounts get a 2x boost.
    """
    score = (
        tweet.get('likes', 0) * 1.0 +
        tweet.get('retweets', 0) * 3.0 +
        tweet.get('replies', 0) * 2.0 +
        tweet.get('quotes', 0) * 4.0
    )
    if tweet.get('is_priority_account'):
        score *= 2.0
    return score


def compute_velocity(current_engagement: float, baseline_engagement: float,
                     current_hours: float, baseline_hours: float) -> float:
    """
    Compute engagement velocity — how fast engagement is accelerating
    relative to the baseline period.
    
    Returns a multiplier: 1.0 = normal, 5.0 = 5x above baseline rate.
    """
    if baseline_hours <= 0 or current_hours <= 0:
        return 0.0

    baseline_rate = baseline_engagement / baseline_hours if baseline_engagement > 0 else 1.0
    current_rate = current_engagement / current_hours

    velocity = current_rate / baseline_rate if baseline_rate > 0 else current_rate
    return round(velocity, 2)


class TrendDetector:
    def __init__(self, config: dict):
        self.baseline_hours = config.get('baseline_window_hours', 24)
        self.trending_hours = config.get('trending_window_hours', 2)
        self.min_velocity = config.get('min_velocity_score', 3.0)
        self.max_trends = config.get('max_trends_per_community', 5)
        self.dedup_threshold = config.get('dedup_similarity_threshold', 0.75)

    def detect_trends(self, current_tweets: list, baseline_tweets: list,
                      community: str) -> list[dict]:
        """
        Detect trending topics by comparing current vs baseline engagement.
        
        Returns list of trend dicts sorted by velocity_score.
        """
        # Extract and count keywords in both windows
        current_keywords = Counter()
        baseline_keywords = Counter()
        current_engagement_by_keyword = defaultdict(float)
        baseline_engagement_by_keyword = defaultdict(float)
        keyword_tweets = defaultdict(list)  # keyword -> list of tweets

        for tweet in current_tweets:
            keywords = extract_keywords(tweet['text'])
            engagement = compute_engagement_score(tweet)
            seen = set()
            for kw in keywords:
                if kw not in seen:
                    current_keywords[kw] += 1
                    current_engagement_by_keyword[kw] += engagement
                    keyword_tweets[kw].append(tweet)
                    seen.add(kw)

        for tweet in baseline_tweets:
            keywords = extract_keywords(tweet['text'])
            engagement = compute_engagement_score(tweet)
            seen = set()
            for kw in keywords:
                if kw not in seen:
                    baseline_keywords[kw] += 1
                    baseline_engagement_by_keyword[kw] += engagement
                    seen.add(kw)

        # Score each keyword by velocity
        keyword_scores = []
        for kw, count in current_keywords.items():
            if count < 3:  # Need at least 3 tweets mentioning it
                continue

            velocity = compute_velocity(
                current_engagement_by_keyword[kw],
                baseline_engagement_by_keyword.get(kw, 0),
                self.trending_hours,
                self.baseline_hours
            )

            if velocity >= self.min_velocity:
                # Get top tweets for this keyword
                kw_tweets = sorted(
                    keyword_tweets[kw],
                    key=lambda t: compute_engagement_score(t),
                    reverse=True
                )[:10]

                keyword_scores.append({
                    'keyword': kw,
                    'velocity': velocity,
                    'tweet_count': count,
                    'total_engagement': current_engagement_by_keyword[kw],
                    'tweets': kw_tweets,
                })

        keyword_scores.sort(key=lambda x: x['velocity'], reverse=True)

        # Cluster similar keywords into topics
        trends = self._cluster_keywords(keyword_scores, community)

        return trends[:self.max_trends]

    def _cluster_keywords(self, keyword_scores: list, community: str) -> list[dict]:
        """
        Cluster related keywords into coherent topics.
        Simple approach: merge keywords that frequently co-occur in the same tweets.
        """
        if not keyword_scores:
            return []

        used = set()
        trends = []

        for item in keyword_scores:
            kw = item['keyword']
            if kw in used:
                continue

            # Find related keywords (co-occurring in same tweets)
            cluster_keywords = [kw]
            cluster_tweets = set(t['tweet_id'] for t in item['tweets'])
            cluster_velocity = item['velocity']
            cluster_engagement = item['total_engagement']

            for other in keyword_scores:
                other_kw = other['keyword']
                if other_kw == kw or other_kw in used:
                    continue
                # Check co-occurrence
                other_tweets = set(t['tweet_id'] for t in other['tweets'])
                overlap = len(cluster_tweets & other_tweets)
                if overlap >= 2 or (overlap >= 1 and len(other_tweets) <= 5):
                    cluster_keywords.append(other_kw)
                    cluster_tweets |= other_tweets
                    cluster_velocity = max(cluster_velocity, other['velocity'])
                    cluster_engagement += other['total_engagement']
                    used.add(other_kw)

            used.add(kw)

            # Build the topic name from top keywords
            topic = " + ".join(cluster_keywords[:3])
            if len(cluster_keywords) > 3:
                topic += f" (+{len(cluster_keywords) - 3} more)"

            # Get top tweet IDs
            all_tweets = []
            for ki in keyword_scores:
                if ki['keyword'] in cluster_keywords:
                    all_tweets.extend(ki['tweets'])

            # Deduplicate and sort
            seen_ids = set()
            unique_tweets = []
            for t in all_tweets:
                if t['tweet_id'] not in seen_ids:
                    seen_ids.add(t['tweet_id'])
                    unique_tweets.append(t)
            unique_tweets.sort(key=lambda t: compute_engagement_score(t), reverse=True)

            trends.append({
                'community': community,
                'topic': topic,
                'keywords': cluster_keywords,
                'velocity_score': cluster_velocity,
                'tweet_count': len(unique_tweets),
                'total_engagement': cluster_engagement,
                'top_tweet_ids': [t['tweet_id'] for t in unique_tweets[:5]],
                'top_tweets': unique_tweets[:5],
            })

        trends.sort(key=lambda t: t['velocity_score'], reverse=True)
        return trends
