"""
Trend detection engine v3 — Semantic Embedding Edition.

Instead of keyword co-occurrence clustering, we:
  1. Clean + dedup RTs
  2. Embed each tweet with a sentence transformer (all-MiniLM-L6-v2)
  3. Cluster with HDBSCAN (auto-detects cluster count)
  4. Label clusters by extracting top entities/phrases
  5. Score by engagement velocity vs baseline
"""

import re
import math
import time
import logging
import numpy as np
from collections import Counter, defaultdict
from typing import Optional

logger = logging.getLogger("twitter_radar.detector")

# Lazy-load heavy models
_model = None

def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading sentence transformer model...")
        _model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Model loaded.")
    return _model


# ──────────────────────────────────────────────────────────────
# STOP WORDS
# ──────────────────────────────────────────────────────────────

STOP_WORDS = {
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
# TEXT CLEANING
# ──────────────────────────────────────────────────────────────

def clean_tweet_text(text: str) -> str:
    """Clean tweet for embedding. Keep semantic content, remove noise."""
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    # Remove RT prefix
    text = re.sub(r'^RT @\w+:\s*', '', text)
    # Keep cashtags/hashtags/mentions as words (remove the symbol)
    text = re.sub(r'[$#@](\w+)', r'\1', text)
    # Remove non-alphanumeric except spaces and basic punctuation
    text = re.sub(r'[^\w\s.,!?\'-]', ' ', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ──────────────────────────────────────────────────────────────
# ENTITY EXTRACTION (for cluster labeling, not clustering)
# ──────────────────────────────────────────────────────────────

def extract_label_entities(text: str) -> list[str]:
    """
    Extract meaningful entities for labeling a cluster.
    Returns a flat list of entities, prioritized.
    """
    entities = []
    clean = re.sub(r'https?://\S+', '', text)

    # Cashtags (highest signal)
    for m in re.findall(r'\$([A-Za-z]{2,6})', clean):
        entities.append(f"${m.upper()}")

    # Hashtags
    for m in re.findall(r'#(\w{2,30})', clean):
        entities.append(f"#{m.lower()}")

    # Mentions
    for m in re.findall(r'@(\w{1,30})', clean):
        entities.append(f"@{m.lower()}")

    # Clean for keyword extraction
    clean = re.sub(r'[#$@]\w+', '', clean)
    clean = re.sub(r'[^a-zA-Z0-9\s\'-]', ' ', clean)
    clean = re.sub(r'\s+', ' ', clean).strip()
    words = clean.lower().split()

    # Bigrams
    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i + 1]
        if w1 not in STOP_WORDS and w2 not in STOP_WORDS and len(w1) > 2 and len(w2) > 2:
            entities.append(f"{w1} {w2}")

    # Significant single words
    for w in words:
        if len(w) > 3 and w not in STOP_WORDS:
            entities.append(w)

    return entities


# ──────────────────────────────────────────────────────────────
# RT HANDLING
# ──────────────────────────────────────────────────────────────

def parse_rt(tweet: dict) -> dict:
    """Extract original text from RTs."""
    text = tweet.get('text', '')
    tweet = dict(tweet)

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
    """Deduplicate RTs — keep highest engagement version."""
    groups = defaultdict(list)
    for t in tweets:
        t = parse_rt(t)
        key = t.get('original_text', t.get('text', ''))[:100].lower().strip()
        groups[key].append(t)

    deduped = []
    for key, group in groups.items():
        best = max(group, key=lambda t: compute_engagement_score(t))
        total_rts = sum(t.get('retweets', 0) for t in group)
        best['retweets'] = max(best.get('retweets', 0), total_rts)
        best['rt_count'] = len(group)
        deduped.append(best)

    return deduped


# ──────────────────────────────────────────────────────────────
# ENGAGEMENT SCORING
# ──────────────────────────────────────────────────────────────

def compute_engagement_score(tweet: dict) -> float:
    """
    Weighted engagement, normalized by author's reach.
    Small accounts with high engagement = strong signal.
    """
    rt_boost = tweet.get('rt_count', 1)

    raw = (
        tweet.get('likes', 0) * 1.0 +
        tweet.get('retweets', 0) * 2.0 +
        tweet.get('replies', 0) * 1.5 +
        tweet.get('quotes', 0) * 3.0
    ) * math.sqrt(rt_boost)

    followers = tweet.get('author_followers', 0)
    if followers > 0:
        rate = raw / math.sqrt(followers)
        return rate * math.log2(max(raw, 2))
    else:
        return raw


def compute_velocity(current_score: float, baseline_score: float,
                     current_hours: float, baseline_hours: float) -> float:
    """
    Engagement velocity — how fast a topic is accelerating.
    No baseline → logarithmic dampening (no more 49,000x on first scan).
    """
    if current_hours <= 0:
        return 0.0

    current_rate = current_score / current_hours

    if baseline_hours <= 0 or baseline_score <= 0:
        # No baseline — log10 dampening. Max ~10x on first scan.
        if current_rate <= 0:
            return 0.0
        return round(min(1.0 + math.log10(max(current_rate, 1)), 10.0), 2)

    baseline_rate = baseline_score / baseline_hours
    if baseline_rate <= 0:
        return min(current_rate, 50.0)

    velocity = current_rate / baseline_rate
    return round(min(velocity, 100.0), 2)


# ──────────────────────────────────────────────────────────────
# TREND DETECTOR
# ──────────────────────────────────────────────────────────────

class TrendDetector:
    def __init__(self, config: dict):
        self.baseline_hours = config.get('baseline_window_hours', 24)
        self.trending_hours = config.get('trending_window_hours', 4)
        self.min_velocity = config.get('min_velocity_score', 2.0)
        self.max_trends = config.get('max_trends_per_community', 10)
        self.min_cluster_size = config.get('min_cluster_size', 5)

    def detect_trends(self, current_tweets: list, baseline_tweets: list,
                      community: str) -> list[dict]:
        """
        Detect trending topics using semantic embedding clustering.

        1. Dedup RTs
        2. Embed tweets with sentence transformer
        3. Cluster with HDBSCAN
        4. Score clusters by engagement velocity
        5. Label clusters by top entities
        6. Return sorted trends
        """
        import hdbscan as hdb

        # Step 1: Dedup RTs
        current_tweets = dedup_rts(current_tweets)
        if baseline_tweets:
            baseline_tweets = dedup_rts(baseline_tweets)

        if len(current_tweets) < self.min_cluster_size:
            logger.info(f"Only {len(current_tweets)} tweets after dedup, skipping clustering")
            return []

        # Step 2: Clean text and compute embeddings
        model = _get_model()
        texts = []
        valid_tweets = []
        for t in current_tweets:
            cleaned = clean_tweet_text(t.get('original_text', t.get('text', '')))
            if len(cleaned.split()) >= 3:  # skip very short tweets
                texts.append(cleaned)
                valid_tweets.append(t)

        if len(texts) < self.min_cluster_size:
            logger.info(f"Only {len(texts)} valid tweets for embedding, skipping")
            return []

        logger.info(f"Embedding {len(texts)} tweets...")
        embeddings = model.encode(texts, show_progress_bar=False, batch_size=64)

        # Step 3: Cluster with HDBSCAN
        # min_cluster_size controls granularity — smaller = more clusters
        clusterer = hdb.HDBSCAN(
            min_cluster_size=max(self.min_cluster_size, 3),
            min_samples=2,
            metric='euclidean',
            cluster_selection_method='eom',  # excess of mass — good for varied densities
        )
        labels = clusterer.fit_predict(embeddings)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        logger.info(f"HDBSCAN found {n_clusters} clusters, {n_noise} noise points")

        # Step 4: Build cluster data
        clusters = defaultdict(list)
        for i, label in enumerate(labels):
            if label == -1:
                continue  # noise point, skip
            clusters[label].append(valid_tweets[i])

        # Step 5: Compute baseline engagement per cluster
        # Embed baseline tweets and assign to nearest cluster centroid
        baseline_cluster_engagement = defaultdict(float)
        if baseline_tweets and n_clusters > 0:
            # Compute cluster centroids
            centroids = {}
            for label, tweets in clusters.items():
                indices = [i for i, l in enumerate(labels) if l == label]
                centroids[label] = np.mean(embeddings[indices], axis=0)

            # Embed baseline tweets and find nearest centroid
            baseline_texts = []
            baseline_valid = []
            for t in baseline_tweets:
                cleaned = clean_tweet_text(t.get('original_text', t.get('text', '')))
                if len(cleaned.split()) >= 3:
                    baseline_texts.append(cleaned)
                    baseline_valid.append(t)

            if baseline_texts:
                logger.info(f"Embedding {len(baseline_texts)} baseline tweets...")
                baseline_embeddings = model.encode(baseline_texts, show_progress_bar=False, batch_size=64)

                centroid_labels = list(centroids.keys())
                centroid_matrix = np.array([centroids[l] for l in centroid_labels])

                for i, emb in enumerate(baseline_embeddings):
                    # Cosine similarity to each centroid
                    sims = np.dot(centroid_matrix, emb) / (
                        np.linalg.norm(centroid_matrix, axis=1) * np.linalg.norm(emb) + 1e-8
                    )
                    best_idx = np.argmax(sims)
                    if sims[best_idx] > 0.5:  # only assign if reasonably similar
                        best_label = centroid_labels[best_idx]
                        baseline_cluster_engagement[best_label] += compute_engagement_score(baseline_valid[i])

        # Step 6: Score and label each cluster
        trends = []
        for label, cluster_tweets in clusters.items():
            if len(cluster_tweets) < self.min_cluster_size:
                continue

            # Total engagement for this cluster
            total_engagement = sum(compute_engagement_score(t) for t in cluster_tweets)

            # Velocity
            velocity = compute_velocity(
                total_engagement,
                baseline_cluster_engagement.get(label, 0),
                self.trending_hours,
                self.baseline_hours
            )

            if velocity < self.min_velocity:
                continue

            # Label: extract entities from all tweets, pick top ones
            entity_counter = Counter()
            for t in cluster_tweets:
                entities = extract_label_entities(t.get('original_text', t.get('text', '')))
                for e in entities:
                    entity_counter[e] += 1

            # Pick top 3 entities that appear in >20% of cluster tweets
            min_freq = max(2, len(cluster_tweets) * 0.2)
            top_entities = [
                e for e, c in entity_counter.most_common(10)
                if c >= min_freq
            ][:3]

            if not top_entities:
                # Fallback: just use the top 3 most common
                top_entities = [e for e, _ in entity_counter.most_common(3)]

            trend_name = " / ".join(
                e.title() if not e.startswith(('$', '#', '@')) and ' ' not in e else e
                for e in top_entities
            )

            # Sort tweets by engagement
            cluster_tweets.sort(key=lambda t: compute_engagement_score(t), reverse=True)

            trends.append({
                'community': community,
                'topic': trend_name,
                'keywords': [e for e, _ in entity_counter.most_common(15)],
                'velocity_score': velocity,
                'tweet_count': len(cluster_tweets),
                'total_engagement': total_engagement,
                'top_tweet_ids': [t['tweet_id'] for t in cluster_tweets[:5]],
                'top_tweets': cluster_tweets[:5],
            })

        # Sort by velocity * log(engagement)
        trends.sort(
            key=lambda t: t['velocity_score'] * math.log2(max(t['total_engagement'], 2)),
            reverse=True
        )

        return trends[:self.max_trends]
