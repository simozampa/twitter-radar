"""
Trend detection engine v2.
Identifies trending topics using:
  - Entity/phrase extraction (not single words)
  - Engagement normalization (relative to author's reach)
  - RT deduplication (credit original, not retweeter)
  - Tighter clustering (co-occurrence threshold)
  - Velocity = acceleration over baseline
"""

import re
import math
import time
import logging
from collections import Counter, defaultdict
from typing import Optional

logger = logging.getLogger("twitter_radar.detector")

# ──────────────────────────────────────────────────────────────
# STOP WORDS (expanded for twitter)
# ──────────────────────────────────────────────────────────────

STOP_WORDS = {
    # English common
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "ought", "used",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
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
    # Twitter-specific noise
    "https", "http", "amp", "rt", "via", "lol", "lmao", "gonna", "gotta",
    "thing", "things", "yeah", "yes", "hey", "let", "take", "every",
    "literally", "actually", "basically", "probably", "definitely",
    "already", "always", "never", "ever", "many", "much", "any",
    "something", "anything", "everything", "nothing", "someone",
    "anyone", "everyone", "lot", "lots", "bit", "long", "big",
    "first", "last", "next", "best", "real", "sure", "hard",
    "put", "keep", "try", "start", "give", "tell", "call",
    "run", "find", "use", "work", "show", "play", "move",
    "live", "feel", "high", "point", "end", "turn", "left",
    "help", "line", "day", "man", "men", "old", "year", "years",
    "today", "week", "month", "world", "life", "part", "while",
    "since", "though", "enough", "goes", "done", "seen",
    "won", "gets", "got", "set", "went", "came", "made",
    "being", "having", "doing", "getting", "making", "going",
    "looking", "coming", "taking", "saying", "thinking",
    "true", "false", "full", "free", "based", "post",
    "read", "watch", "check", "follow", "share", "drop",
    "claim", "believe", "mean", "stop",
}


# ──────────────────────────────────────────────────────────────
# ENTITY / PHRASE EXTRACTION
# ──────────────────────────────────────────────────────────────

def extract_entities(text: str) -> dict:
    """
    Extract meaningful entities from a tweet.
    Returns dict with categorized entities.
    """
    entities = {
        'cashtags': [],     # $BTC, $ETH
        'hashtags': [],     # #AI, #crypto
        'mentions': [],     # @OpenAI
        'phrases': [],      # multi-word phrases
        'keywords': [],     # significant single words
    }

    # Remove URLs
    clean = re.sub(r'https?://\S+', '', text)

    # Extract cashtags (high signal)
    entities['cashtags'] = [m.upper() for m in re.findall(r'\$([A-Za-z]{2,6})', clean)]

    # Extract hashtags
    entities['hashtags'] = [m.lower() for m in re.findall(r'#(\w{2,30})', clean)]

    # Extract mentions
    entities['mentions'] = [m.lower() for m in re.findall(r'@(\w{1,30})', clean)]

    # Clean text for phrase extraction
    clean = re.sub(r'[#$@]\w+', '', clean)  # remove tags
    clean = re.sub(r'[^a-zA-Z0-9\s\'-]', ' ', clean)
    clean = re.sub(r'\s+', ' ', clean).strip()
    words = clean.lower().split()

    # Extract bigrams (two-word phrases)
    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i + 1]
        if w1 not in STOP_WORDS and w2 not in STOP_WORDS and len(w1) > 2 and len(w2) > 2:
            entities['phrases'].append(f"{w1} {w2}")

    # Extract significant single words (proper nouns, longer words)
    for w in words:
        if len(w) > 3 and w not in STOP_WORDS:
            entities['keywords'].append(w)

    return entities


def get_all_topics(entities: dict) -> list[str]:
    """Get a flat list of all topics from entities, prioritized."""
    topics = []
    # Priority order: cashtags > hashtags > phrases > keywords
    topics.extend([f"${t}" for t in entities['cashtags']])
    topics.extend([f"#{t}" for t in entities['hashtags']])
    topics.extend(entities['phrases'])
    topics.extend(entities['keywords'])
    return topics


# ──────────────────────────────────────────────────────────────
# RT HANDLING
# ──────────────────────────────────────────────────────────────

def parse_rt(tweet: dict) -> dict:
    """
    If tweet is a RT, extract the original author and text.
    Returns modified tweet dict with rt_author and cleaned text.
    """
    text = tweet.get('text', '')
    tweet = dict(tweet)  # don't mutate original

    if text.startswith('RT @'):
        match = re.match(r'^RT @(\w+):\s*(.*)', text, re.DOTALL)
        if match:
            tweet['rt_author'] = match.group(1).lower()
            tweet['original_text'] = match.group(2)
            tweet['is_rt'] = True
            return tweet

    tweet['is_rt'] = False
    tweet['original_text'] = text
    return tweet


def dedup_rts(tweets: list) -> list:
    """
    Deduplicate RTs — keep the version with highest engagement.
    Group by original text similarity.
    """
    # Group by first 100 chars of original text
    groups = defaultdict(list)
    for t in tweets:
        t = parse_rt(t)
        key = t.get('original_text', t.get('text', ''))[:100].lower().strip()
        groups[key].append(t)

    deduped = []
    for key, group in groups.items():
        # Keep the one with highest engagement
        best = max(group, key=lambda t: compute_engagement_score(t))
        # But aggregate RT count from all versions
        total_rts = sum(t.get('retweets', 0) for t in group)
        best['retweets'] = max(best.get('retweets', 0), total_rts)
        best['rt_count'] = len(group)  # how many people RTed this
        deduped.append(best)

    return deduped


# ──────────────────────────────────────────────────────────────
# ENGAGEMENT SCORING
# ──────────────────────────────────────────────────────────────

def compute_engagement_score(tweet: dict) -> float:
    """
    Weighted engagement score.
    Normalized by author's reach (follower count).
    A 100-like tweet from a 1k-follower account is hotter
    than a 100-like tweet from a 10M-follower account.
    """
    # RT count (how many people in your feed RTed this) is strong signal
    rt_boost = tweet.get('rt_count', 1)

    raw = (
        tweet.get('likes', 0) * 1.0 +
        tweet.get('retweets', 0) * 2.0 +
        tweet.get('replies', 0) * 1.5 +
        tweet.get('quotes', 0) * 3.0
    ) * math.sqrt(rt_boost)  # boost by how viral it went in your feed

    # Normalize by follower count (engagement rate)
    followers = tweet.get('author_followers', 0)
    if followers > 0:
        # Engagement rate with diminishing returns for huge accounts
        rate = raw / math.sqrt(followers)
        # Boost: small accounts with high engagement = strong signal
        return rate * math.log2(max(raw, 2))
    else:
        return raw


def compute_velocity(current_score: float, baseline_score: float,
                     current_hours: float, baseline_hours: float) -> float:
    """
    Engagement velocity — how fast a topic is accelerating.
    Returns multiplier: 1.0 = normal, 5.0 = 5x above baseline.
    Handles edge cases (no baseline = moderate velocity, not infinite).
    """
    if current_hours <= 0:
        return 0.0

    current_rate = current_score / current_hours

    if baseline_hours <= 0 or baseline_score <= 0:
        # No baseline — assign moderate velocity based on raw engagement
        # This prevents "everything is infinity" on first scan
        if current_rate > 100:
            return min(current_rate / 10, 50.0)  # cap at 50x
        elif current_rate > 10:
            return min(current_rate / 5, 20.0)
        else:
            return min(current_rate, 10.0)

    baseline_rate = baseline_score / baseline_hours
    if baseline_rate <= 0:
        return min(current_rate, 50.0)

    velocity = current_rate / baseline_rate
    return round(min(velocity, 100.0), 2)  # cap at 100x


# ──────────────────────────────────────────────────────────────
# TREND DETECTOR
# ──────────────────────────────────────────────────────────────

class TrendDetector:
    def __init__(self, config: dict):
        self.baseline_hours = config.get('baseline_window_hours', 24)
        self.trending_hours = config.get('trending_window_hours', 4)
        self.min_velocity = config.get('min_velocity_score', 2.0)
        self.max_trends = config.get('max_trends_per_community', 5)
        self.dedup_threshold = config.get('dedup_similarity_threshold', 0.75)

    def detect_trends(self, current_tweets: list, baseline_tweets: list,
                      community: str) -> list[dict]:
        """
        Detect trending topics.
        1. Dedup RTs
        2. Extract entities/phrases
        3. Count topic frequency + engagement
        4. Compute velocity vs baseline
        5. Cluster related topics
        6. Return sorted trends
        """
        # Step 1: Dedup RTs
        current_tweets = dedup_rts(current_tweets)
        if baseline_tweets:
            baseline_tweets = dedup_rts(baseline_tweets)

        # Step 2: Extract topics and score
        topic_data = defaultdict(lambda: {
            'count': 0,
            'engagement': 0.0,
            'tweets': [],
            'type': 'keyword',  # cashtag, hashtag, phrase, keyword
        })

        for tweet in current_tweets:
            entities = extract_entities(tweet.get('original_text', tweet.get('text', '')))
            engagement = compute_engagement_score(tweet)
            seen_topics = set()

            # Process each entity type with priority weighting
            for cashtag in entities['cashtags']:
                topic = f"${cashtag}"
                if topic not in seen_topics:
                    topic_data[topic]['count'] += 1
                    topic_data[topic]['engagement'] += engagement * 2.0  # boost cashtags
                    topic_data[topic]['tweets'].append(tweet)
                    topic_data[topic]['type'] = 'cashtag'
                    seen_topics.add(topic)

            for hashtag in entities['hashtags']:
                topic = f"#{hashtag}"
                if topic not in seen_topics:
                    topic_data[topic]['count'] += 1
                    topic_data[topic]['engagement'] += engagement * 1.5
                    topic_data[topic]['tweets'].append(tweet)
                    topic_data[topic]['type'] = 'hashtag'
                    seen_topics.add(topic)

            for phrase in entities['phrases']:
                if phrase not in seen_topics:
                    topic_data[phrase]['count'] += 1
                    topic_data[phrase]['engagement'] += engagement * 1.5  # phrases > single words
                    topic_data[phrase]['tweets'].append(tweet)
                    topic_data[phrase]['type'] = 'phrase'
                    seen_topics.add(phrase)

            for keyword in entities['keywords']:
                if keyword not in seen_topics:
                    topic_data[keyword]['count'] += 1
                    topic_data[keyword]['engagement'] += engagement
                    topic_data[keyword]['tweets'].append(tweet)
                    topic_data[keyword]['type'] = 'keyword'
                    seen_topics.add(keyword)

        # Step 3: Baseline engagement per topic
        baseline_engagement = defaultdict(float)
        for tweet in (baseline_tweets or []):
            entities = extract_entities(tweet.get('original_text', tweet.get('text', '')))
            engagement = compute_engagement_score(tweet)
            for topic in get_all_topics(entities):
                baseline_engagement[topic] += engagement

        # Step 4: Score and filter topics
        scored_topics = []
        for topic, data in topic_data.items():
            # Higher threshold for single keywords (noisy), lower for entities/phrases
            min_count = 2
            if data['type'] == 'keyword':
                min_count = 3  # single words need more evidence
            if data['count'] < min_count:
                continue

            velocity = compute_velocity(
                data['engagement'],
                baseline_engagement.get(topic, 0),
                self.trending_hours,
                self.baseline_hours
            )

            if velocity < self.min_velocity:
                continue

            # Sort tweets by engagement
            data['tweets'].sort(key=lambda t: compute_engagement_score(t), reverse=True)

            scored_topics.append({
                'topic': topic,
                'type': data['type'],
                'velocity': velocity,
                'count': data['count'],
                'engagement': data['engagement'],
                'tweets': data['tweets'][:10],
            })

        # Sort by velocity * engagement (both matter)
        scored_topics.sort(
            key=lambda x: x['velocity'] * math.log2(max(x['engagement'], 2)),
            reverse=True
        )

        # Step 5: Cluster related topics
        trends = self._cluster_topics(scored_topics, community)

        return trends[:self.max_trends]

    def _cluster_topics(self, scored_topics: list, community: str) -> list[dict]:
        """
        Cluster related topics into coherent trends.
        Uses tweet overlap — if topics share many of the same tweets,
        they're part of the same trend.
        """
        if not scored_topics:
            return []

        used = set()
        trends = []

        for item in scored_topics:
            topic = item['topic']
            if topic in used:
                continue

            # Start a new cluster
            cluster_topics = [topic]
            cluster_tweet_ids = set(t['tweet_id'] for t in item['tweets'])
            cluster_engagement = item['engagement']
            cluster_velocity = item['velocity']
            cluster_type = item['type']

            # Find related topics (high tweet overlap)
            for other in scored_topics:
                other_topic = other['topic']
                if other_topic == topic or other_topic in used:
                    continue

                other_tweet_ids = set(t['tweet_id'] for t in other['tweets'])
                overlap = len(cluster_tweet_ids & other_tweet_ids)
                min_size = min(len(cluster_tweet_ids), len(other_tweet_ids))

                # Require >25% overlap to merge (looser = bigger clusters)
                if min_size > 0 and overlap / min_size > 0.25:
                    cluster_topics.append(other_topic)
                    cluster_tweet_ids |= other_tweet_ids
                    cluster_engagement += other['engagement']
                    cluster_velocity = max(cluster_velocity, other['velocity'])
                    used.add(other_topic)

            used.add(topic)

            # Build trend name — prioritize by type and engagement
            # Score each topic by: type priority + engagement contribution
            type_priority = {'cashtag': 4, 'hashtag': 3, 'phrase': 2, 'keyword': 1}
            topic_scores = []
            for t in cluster_topics:
                # Find this topic's data
                for st in scored_topics:
                    if st['topic'] == t:
                        score = (
                            type_priority.get(st['type'], 0) * 100 +
                            st['engagement']
                        )
                        topic_scores.append((t, score, st['type']))
                        break
                else:
                    topic_scores.append((t, 0, 'keyword'))

            topic_scores.sort(key=lambda x: x[1], reverse=True)
            name_parts = []
            for t, _, ttype in topic_scores[:3]:
                if t.startswith('$') or t.startswith('#'):
                    name_parts.append(t)
                elif ' ' in t:
                    name_parts.append(t.title())
                else:
                    name_parts.append(t.title())

            trend_name = " / ".join(name_parts)
            if len(cluster_topics) > 3:
                trend_name += f" (+{len(cluster_topics) - 3})"

            # Collect all unique tweets, sorted by engagement
            all_tweets = []
            seen_ids = set()
            for st in scored_topics:
                if st['topic'] in cluster_topics or st['topic'] == topic:
                    for t in st['tweets']:
                        if t['tweet_id'] not in seen_ids:
                            seen_ids.add(t['tweet_id'])
                            all_tweets.append(t)
            all_tweets.sort(key=lambda t: compute_engagement_score(t), reverse=True)

            trends.append({
                'community': community,
                'topic': trend_name,
                'keywords': cluster_topics,
                'velocity_score': cluster_velocity,
                'tweet_count': len(all_tweets),
                'total_engagement': cluster_engagement,
                'top_tweet_ids': [t['tweet_id'] for t in all_tweets[:5]],
                'top_tweets': all_tweets[:5],
            })

        trends.sort(key=lambda t: t['velocity_score'] * math.log2(max(t['total_engagement'], 2)),
                     reverse=True)
        return trends
