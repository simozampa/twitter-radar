"""
Database layer for Twitter Trend Radar.
SQLite-based storage for tweets, trends, and alert history.
"""

import sqlite3
import json
import time
from pathlib import Path
from contextlib import contextmanager
from typing import Optional

DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS tweets (
    tweet_id TEXT PRIMARY KEY,
    author_id TEXT NOT NULL,
    author_username TEXT,
    text TEXT NOT NULL,
    community TEXT NOT NULL,
    created_at TEXT NOT NULL,
    likes INTEGER DEFAULT 0,
    retweets INTEGER DEFAULT 0,
    replies INTEGER DEFAULT 0,
    quotes INTEGER DEFAULT 0,
    impressions INTEGER DEFAULT 0,
    is_priority_account BOOLEAN DEFAULT 0,
    fetched_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS trends (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    community TEXT NOT NULL,
    topic TEXT NOT NULL,
    keywords TEXT NOT NULL,  -- JSON array
    velocity_score REAL NOT NULL,
    tweet_count INTEGER NOT NULL,
    total_engagement INTEGER NOT NULL,
    top_tweet_ids TEXT NOT NULL,  -- JSON array
    detected_at REAL NOT NULL,
    expires_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trend_id INTEGER,
    community TEXT NOT NULL,
    topic TEXT NOT NULL,
    velocity_score REAL NOT NULL,
    drafts TEXT,  -- JSON array of generated drafts
    sent_at REAL NOT NULL,
    FOREIGN KEY (trend_id) REFERENCES trends(id)
);

CREATE TABLE IF NOT EXISTS fetch_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    community TEXT NOT NULL,
    query TEXT NOT NULL,
    tweet_count INTEGER NOT NULL,
    fetched_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tweets_community ON tweets(community, created_at);
CREATE INDEX IF NOT EXISTS idx_tweets_fetched ON tweets(fetched_at);
CREATE INDEX IF NOT EXISTS idx_trends_community ON trends(community, detected_at);
CREATE INDEX IF NOT EXISTS idx_alerts_sent ON alerts(sent_at);
"""


class Database:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript(DB_SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def upsert_tweet(self, tweet: dict):
        """Insert or update a tweet."""
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO tweets (tweet_id, author_id, author_username, text, community,
                                    created_at, likes, retweets, replies, quotes,
                                    impressions, is_priority_account, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tweet_id) DO UPDATE SET
                    likes=excluded.likes,
                    retweets=excluded.retweets,
                    replies=excluded.replies,
                    quotes=excluded.quotes,
                    impressions=excluded.impressions,
                    fetched_at=excluded.fetched_at
            """, (
                tweet['tweet_id'], tweet['author_id'], tweet.get('author_username'),
                tweet['text'], tweet['community'], tweet['created_at'],
                tweet.get('likes', 0), tweet.get('retweets', 0),
                tweet.get('replies', 0), tweet.get('quotes', 0),
                tweet.get('impressions', 0), tweet.get('is_priority_account', False),
                time.time()
            ))

    def upsert_tweets_batch(self, tweets: list):
        """Batch insert/update tweets."""
        with self._conn() as conn:
            conn.executemany("""
                INSERT INTO tweets (tweet_id, author_id, author_username, text, community,
                                    created_at, likes, retweets, replies, quotes,
                                    impressions, is_priority_account, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tweet_id) DO UPDATE SET
                    likes=excluded.likes,
                    retweets=excluded.retweets,
                    replies=excluded.replies,
                    quotes=excluded.quotes,
                    impressions=excluded.impressions,
                    fetched_at=excluded.fetched_at
            """, [(
                t['tweet_id'], t['author_id'], t.get('author_username'),
                t['text'], t['community'], t['created_at'],
                t.get('likes', 0), t.get('retweets', 0),
                t.get('replies', 0), t.get('quotes', 0),
                t.get('impressions', 0), t.get('is_priority_account', False),
                time.time()
            ) for t in tweets])

    def get_tweets_in_window(self, community: str, start_time: float, end_time: Optional[float] = None) -> list:
        """Get tweets within a time window for a community."""
        end_time = end_time or time.time()
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT * FROM tweets
                WHERE community = ? AND fetched_at >= ? AND fetched_at <= ?
                ORDER BY (likes + retweets * 2 + quotes * 3) DESC
            """, (community, start_time, end_time)).fetchall()
            return [dict(r) for r in rows]

    def get_top_tweets(self, community: str, hours: float, limit: int = 50) -> list:
        """Get top tweets by engagement in the last N hours."""
        since = time.time() - (hours * 3600)
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT * FROM tweets
                WHERE community = ? AND fetched_at >= ?
                ORDER BY (likes + retweets * 2 + replies + quotes * 3) DESC
                LIMIT ?
            """, (community, since, limit)).fetchall()
            return [dict(r) for r in rows]

    def insert_trend(self, trend: dict) -> int:
        """Insert a detected trend, return its ID."""
        with self._conn() as conn:
            cursor = conn.execute("""
                INSERT INTO trends (community, topic, keywords, velocity_score,
                                    tweet_count, total_engagement, top_tweet_ids,
                                    detected_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trend['community'], trend['topic'],
                json.dumps(trend['keywords']), trend['velocity_score'],
                trend['tweet_count'], trend['total_engagement'],
                json.dumps(trend['top_tweet_ids']),
                time.time(), time.time() + 3600  # 1hr expiry default
            ))
            return cursor.lastrowid

    def insert_alert(self, alert: dict):
        """Log a sent alert."""
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO alerts (trend_id, community, topic, velocity_score, drafts, sent_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                alert.get('trend_id'), alert['community'], alert['topic'],
                alert['velocity_score'], json.dumps(alert.get('drafts', [])),
                time.time()
            ))

    def was_alerted_recently(self, topic: str, community: str, hours: float = 2) -> bool:
        """Check if we already alerted on this topic recently."""
        since = time.time() - (hours * 3600)
        with self._conn() as conn:
            row = conn.execute("""
                SELECT COUNT(*) as cnt FROM alerts
                WHERE topic = ? AND community = ? AND sent_at >= ?
            """, (topic, community, since)).fetchone()
            return row['cnt'] > 0

    def log_fetch(self, community: str, query: str, tweet_count: int):
        """Log a fetch operation."""
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO fetch_log (community, query, tweet_count, fetched_at)
                VALUES (?, ?, ?, ?)
            """, (community, query, tweet_count, time.time()))

    def cleanup_old_data(self, retention_days: int = 30):
        """Remove data older than retention period."""
        cutoff = time.time() - (retention_days * 86400)
        with self._conn() as conn:
            conn.execute("DELETE FROM tweets WHERE fetched_at < ?", (cutoff,))
            conn.execute("DELETE FROM trends WHERE detected_at < ?", (cutoff,))
            conn.execute("DELETE FROM fetch_log WHERE fetched_at < ?", (cutoff,))
