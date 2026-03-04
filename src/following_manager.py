"""
Following list manager.
Categorizes accounts and selects batches for scraping.
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("twitter_radar.following")

# Keyword-based auto-categorization
CATEGORY_KEYWORDS = {
    "ai": {
        "bio_keywords": [
            "ai", "artificial intelligence", "machine learning", "deep learning",
            "neural", "llm", "gpt", "nlp", "computer vision", "robotics",
            "openai", "anthropic", "deepmind", "research", "agi",
            "transformer", "diffusion", "generative",
        ],
        "username_keywords": [
            "ai", "ml", "gpt", "neural", "deep", "llm",
        ],
    },
    "crypto": {
        "bio_keywords": [
            "crypto", "bitcoin", "btc", "ethereum", "eth", "solana", "sol",
            "defi", "web3", "blockchain", "nft", "dex", "dao", "token",
            "trading", "trader", "onchain", "on-chain", "memecoin",
            "coinbase", "binance", "degen", "ape", "airdrop",
        ],
        "username_keywords": [
            "crypto", "btc", "eth", "sol", "defi", "degen", "nft",
            "web3", "chain", "coin",
        ],
    },
    "tech": {
        "bio_keywords": [
            "founder", "ceo", "cto", "startup", "engineer", "developer",
            "software", "saas", "venture", "vc", "yc", "product",
            "open source", "programming", "code", "build",
        ],
        "username_keywords": [
            "dev", "eng", "hack", "code", "build",
        ],
    },
}


class FollowingManager:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.following_path = self.data_dir / "following.json"
        self.categories_path = self.data_dir / "following_categories.json"
        self.following: list = []
        self.categories: dict = {}
        self._load()

    def _load(self):
        """Load following list and categories."""
        if self.following_path.exists():
            with open(self.following_path) as f:
                self.following = json.load(f)

        if self.categories_path.exists():
            with open(self.categories_path) as f:
                self.categories = json.load(f)

        if self.following and not self.categories:
            self._auto_categorize()

    def _auto_categorize(self):
        """Auto-categorize accounts based on bio and username keywords."""
        self.categories = {cat: [] for cat in CATEGORY_KEYWORDS}
        self.categories["uncategorized"] = []

        for account in self.following:
            username = (account.get("username") or "").lower()
            bio = (account.get("description") or "").lower()
            followers = account.get("followers", 0)

            matched_categories = []

            for cat, rules in CATEGORY_KEYWORDS.items():
                bio_match = any(kw in bio for kw in rules["bio_keywords"])
                name_match = any(kw in username for kw in rules["username_keywords"])

                if bio_match or name_match:
                    matched_categories.append(cat)

            if matched_categories:
                for cat in matched_categories:
                    self.categories[cat].append(username)
            else:
                self.categories["uncategorized"].append(username)

        self._save_categories()

        # Log summary
        for cat, usernames in self.categories.items():
            if usernames:
                logger.info(f"Category '{cat}': {len(usernames)} accounts")

    def _save_categories(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with open(self.categories_path, "w") as f:
            json.dump(self.categories, f, indent=2)

    def get_accounts_for_community(self, community: str) -> list[str]:
        """Get usernames for a community."""
        return self.categories.get(community, [])

    def get_all_signal_accounts(self, min_followers: int = 5000,
                                 max_accounts: int = 200) -> list[str]:
        """
        Get high-signal accounts across all categories.
        Sorted by follower count, capped at max_accounts.
        """
        account_map = {a.get("username", "").lower(): a for a in self.following}

        all_categorized = set()
        for cat, usernames in self.categories.items():
            if cat != "uncategorized":
                all_categorized.update(usernames)

        # Filter by min followers and sort
        signal = []
        for username in all_categorized:
            account = account_map.get(username, {})
            followers = account.get("followers", 0)
            if followers >= min_followers:
                signal.append((username, followers))

        signal.sort(key=lambda x: x[1], reverse=True)
        return [u for u, _ in signal[:max_accounts]]

    def get_batch_for_scraping(self, community: str, batch_size: int = 50,
                                rotation_index: int = 0) -> list[str]:
        """
        Get a rotated batch of accounts for scraping.
        Cycles through the community's accounts over multiple scans.
        """
        accounts = self.get_accounts_for_community(community)
        if not accounts:
            return []

        # Sort by follower count for consistent ordering
        account_map = {a.get("username", "").lower(): a for a in self.following}
        accounts_with_followers = [
            (u, account_map.get(u, {}).get("followers", 0))
            for u in accounts
        ]
        accounts_with_followers.sort(key=lambda x: x[1], reverse=True)
        sorted_accounts = [u for u, _ in accounts_with_followers]

        # Rotate
        start = (rotation_index * batch_size) % len(sorted_accounts) if sorted_accounts else 0
        batch = sorted_accounts[start:start + batch_size]

        # If we wrapped around, fill from the beginning
        if len(batch) < batch_size:
            remaining = batch_size - len(batch)
            batch.extend(sorted_accounts[:remaining])

        return batch

    def summary(self) -> str:
        """Return a summary of the following categorization."""
        lines = [f"Total following: {len(self.following)}"]
        for cat, usernames in sorted(self.categories.items()):
            if usernames:
                lines.append(f"  {cat}: {len(usernames)} accounts")
        return "\n".join(lines)
