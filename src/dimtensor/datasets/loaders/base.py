"""Base classes for dataset loaders with caching support."""

from __future__ import annotations

import hashlib
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ..cache import CacheManager, get_cache_manager

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


def get_cache_dir() -> Path:
    """Get the cache directory for dataset downloads.

    Returns:
        Path to cache directory (~/.dimtensor/cache/).

    The cache directory can be overridden with DIMTENSOR_CACHE_DIR
    environment variable.
    """
    cache_dir = os.environ.get("DIMTENSOR_CACHE_DIR")
    if cache_dir:
        return Path(cache_dir)

    # Default to ~/.dimtensor/cache/
    home = Path.home()
    return home / ".dimtensor" / "cache"


def ensure_cache_dir() -> Path:
    """Ensure cache directory exists and return the path."""
    cache_dir = get_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


class BaseLoader(ABC):
    """Base class for dataset loaders with caching support.

    Provides common functionality for downloading, caching, and
    loading physics datasets.

    Attributes:
        cache_enabled: Whether to use caching (default: True).
        cache_dir: Directory for cached files.
        cache_manager: CacheManager instance for metadata tracking.
        source_name: Name of the data source (for cache tracking).
    """

    def __init__(
        self,
        cache: bool = True,
        cache_manager: CacheManager | None = None,
        source_name: str = "",
    ):
        """Initialize the loader.

        Args:
            cache: Whether to enable caching.
            cache_manager: CacheManager instance (default: global manager).
            source_name: Name of this data source for cache tracking.
        """
        self.cache_enabled = cache
        self.cache_manager = cache_manager or get_cache_manager()
        self.cache_dir = self.cache_manager.cache_dir
        self.source_name = source_name or self.__class__.__name__

    @abstractmethod
    def load(self, **kwargs: Any) -> Any:
        """Load the dataset.

        Args:
            **kwargs: Loader-specific arguments.

        Returns:
            The loaded dataset (typically dict of DimArrays).
        """
        pass

    def download(
        self,
        url: str,
        cache_key: str | None = None,
        force: bool = False,
        ttl: float | None = None,
    ) -> Path:
        """Download a file with caching.

        Args:
            url: URL to download from.
            cache_key: Cache identifier (default: hash of URL).
            force: Force re-download even if cached.
            ttl: Time-to-live in seconds (None = use manager default).

        Returns:
            Path to the downloaded (or cached) file.

        Raises:
            ImportError: If requests library not available.
            RuntimeError: If download fails.
        """
        if not HAS_REQUESTS:
            raise ImportError(
                "requests library required for dataset downloads. "
                "Install with: pip install requests"
            )

        # Generate cache key from URL if not provided. MD5 is used as a
        # filename hash, not for security — a collision just means a cache
        # miss. usedforsecurity=False signals that intent to scanners.
        if cache_key is None:
            cache_key = hashlib.md5(url.encode(), usedforsecurity=False).hexdigest()

        # Determine file extension from URL
        parsed = urlparse(url)
        path_parts = parsed.path.split("/")
        if path_parts and "." in path_parts[-1]:
            ext = "." + path_parts[-1].split(".")[-1]
        else:
            ext = ".dat"

        cache_file = self.cache_dir / f"{cache_key}{ext}"

        # Check cache manager for existing entry
        if self.cache_enabled and not force:
            entry = self.cache_manager.get_entry(cache_key)
            if entry is not None and not entry.is_expired():
                # Check if file still exists
                if cache_file.exists():
                    return cache_file
                # File was deleted, remove stale entry
                self.cache_manager.remove_entry(cache_key)

        # Download the file
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to download {url}: {e}") from e

        # Write to cache
        cache_file.write_bytes(response.content)

        # Register with cache manager
        if self.cache_enabled:
            self.cache_manager.add_entry(
                cache_key=cache_key,
                url=url,
                filepath=cache_file,
                ttl=ttl,
                content_type=response.headers.get("content-type", ""),
                source=self.source_name,
            )

        return cache_file

    def get_cache_info(self, cache_key: str) -> dict[str, Any] | None:
        """Get metadata about a cached file.

        Args:
            cache_key: Cache identifier.

        Returns:
            Metadata dict or None if not cached.
        """
        entry = self.cache_manager.get_entry(cache_key)
        if entry is None:
            return None

        from dataclasses import asdict
        return asdict(entry)

    def clear_cache(self, cache_key: str | None = None) -> None:
        """Clear cached files.

        Args:
            cache_key: Specific cache key to clear, or None to clear all from this source.
        """
        if cache_key is None:
            # Clear all cache files from this source
            self.cache_manager.clean_by_source(self.source_name)
        else:
            # Clear specific cache key
            self.cache_manager.remove_entry(cache_key)


class CSVLoader(BaseLoader):
    """Loader for CSV-based datasets.

    Provides utilities for loading and parsing CSV data with
    proper dimensional units.
    """

    def load(self, **kwargs: Any) -> Any:
        """Load CSV dataset.

        Subclasses should override this method with specific
        parsing logic.
        """
        raise NotImplementedError(
            "CSVLoader is a base class. Use a specific loader."
        )

    def parse_csv(
        self,
        filepath: Path,
        skip_rows: int = 0,
        delimiter: str = ",",
    ) -> list[list[str]]:
        """Parse CSV file into rows.

        Args:
            filepath: Path to CSV file.
            skip_rows: Number of header rows to skip.
            delimiter: Column delimiter (default: comma).

        Returns:
            List of rows, each row is a list of string values.
        """
        lines = filepath.read_text().strip().split("\n")

        if skip_rows > 0:
            lines = lines[skip_rows:]

        rows = []
        for line in lines:
            if not line.strip():
                continue
            rows.append([col.strip() for col in line.split(delimiter)])

        return rows
