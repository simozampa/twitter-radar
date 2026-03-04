# Twitter Trend Radar 🔥

Real-time Twitter/X trend detection engine that monitors configurable communities, detects engagement velocity spikes, and generates draft content for trending topics.

## What It Does

- **Monitors communities** — track AI, crypto, tech, or any custom topic with configurable search queries and priority accounts
- **Detects trends by velocity** — not just what's popular, but what's *accelerating*. Compares engagement in a short window vs. a longer baseline to find topics blowing up right now
- **Clusters keywords into topics** — groups related keywords that co-occur in tweets into coherent trending topics
- **Generates draft content** — produces prompts for hot takes, Twitter threads, and article outlines based on detected trends
- **Deduplicates alerts** — won't spam you with the same trend twice
- **Respects quiet hours** — configurable silence window with override threshold for truly explosive trends

## Architecture

```
twitter_client.py   → Twitter API v2 wrapper with rate limiting
trend_detector.py   → Velocity-based trend detection + keyword clustering
draft_generator.py  → Content draft prompt generation
radar.py            → Main orchestrator (fetch → detect → format)
db.py               → SQLite storage (tweets, trends, alerts, fetch logs)
config.yaml         → Full configuration (communities, thresholds, styles)
run_scan.py         → CLI entry point
```

## Setup

### Requirements

- Python 3.10+
- Twitter API Pro plan ($200/mo) — needs v2 search endpoint access
- `requests` and `pyyaml`

```bash
pip install requests pyyaml
```

### Configuration

1. Set your Twitter API bearer token:
```bash
export TWITTER_BEARER_TOKEN="your_token_here"
```

2. Edit `config.yaml` to configure:
   - **Communities** — topics to track, search queries, priority accounts
   - **Detection** — baseline window, trending window, velocity thresholds
   - **Alerts** — digest frequency, quiet hours, draft generation
   - **Storage** — database path, retention period

## Usage

```bash
# Full scan — fetch tweets, detect trends, print results
python run_scan.py

# JSON output (for programmatic use / cron integration)
python run_scan.py --json

# Scan specific community only
python run_scan.py --community crypto

# Verbose logging
python run_scan.py -v

# Dry run (fetch + detect but don't log alerts)
python run_scan.py --dry-run
```

## How Trend Detection Works

1. **Fetch** — pulls recent tweets matching community search queries via Twitter API v2
2. **Baseline** — calculates average engagement rate over a 24-hour window
3. **Current** — measures engagement in the last 2 hours
4. **Velocity** — computes the ratio: `current_rate / baseline_rate`. A velocity of 5.0 means engagement is running 5x above normal
5. **Clustering** — groups co-occurring keywords into coherent topics
6. **Scoring** — weights engagement (quotes 4x, retweets 3x, replies 2x, likes 1x) with a 2x boost for priority accounts

## Configuration Reference

### Communities

```yaml
communities:
  my_topic:
    name: "Display Name"
    enabled: true
    queries:
      - "(keyword1 OR keyword2) -is:retweet lang:en"
    priority_accounts:
      - "importantperson"
    min_likes: 50
    min_retweets: 10
```

### Detection Thresholds

| Setting | Default | Description |
|---------|---------|-------------|
| `baseline_window_hours` | 24 | Lookback for baseline engagement |
| `trending_window_hours` | 2 | Window for "current" activity |
| `min_velocity_score` | 3.0 | Minimum velocity to flag as trend |
| `max_trends_per_community` | 5 | Cap on trends per community |
| `alert_velocity_threshold` | 5.0 | Minimum velocity to trigger alert |

## License

MIT
