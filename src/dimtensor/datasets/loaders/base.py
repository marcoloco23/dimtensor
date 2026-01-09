"""Base classes for dataset loaders with caching support."""

from __future__ import annotations

import hashlib
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

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
    """

    def __init__(self, cache: bool = True):
        """Initialize the loader.

        Args:
            cache: Whether to enable caching.
        """
        self.cache_enabled = cache
        self.cache_dir = ensure_cache_dir()

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
    ) -> Path:
        """Download a file with caching.

        Args:
            url: URL to download from.
            cache_key: Cache identifier (default: hash of URL).
            force: Force re-download even if cached.

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

        # Generate cache key from URL if not provided
        if cache_key is None:
            cache_key = hashlib.md5(url.encode()).hexdigest()

        # Determine file extension from URL
        parsed = urlparse(url)
        path_parts = parsed.path.split("/")
        if path_parts and "." in path_parts[-1]:
            ext = "." + path_parts[-1].split(".")[-1]
        else:
            ext = ".dat"

        cache_file = self.cache_dir / f"{cache_key}{ext}"
        metadata_file = self.cache_dir / f"{cache_key}.json"

        # Check cache
        if self.cache_enabled and not force and cache_file.exists():
            return cache_file

        # Download the file
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to download {url}: {e}") from e

        # Write to cache
        cache_file.write_bytes(response.content)

        # Write metadata
        metadata = {
            "url": url,
            "cache_key": cache_key,
            "size": len(response.content),
            "content_type": response.headers.get("content-type", ""),
        }
        metadata_file.write_text(json.dumps(metadata, indent=2))

        return cache_file

    def get_cache_info(self, cache_key: str) -> dict[str, Any] | None:
        """Get metadata about a cached file.

        Args:
            cache_key: Cache identifier.

        Returns:
            Metadata dict or None if not cached.
        """
        metadata_file = self.cache_dir / f"{cache_key}.json"
        if not metadata_file.exists():
            return None

        return json.loads(metadata_file.read_text())

    def clear_cache(self, cache_key: str | None = None) -> None:
        """Clear cached files.

        Args:
            cache_key: Specific cache key to clear, or None to clear all.
        """
        if cache_key is None:
            # Clear all cache files
            for file in self.cache_dir.iterdir():
                file.unlink()
        else:
            # Clear specific cache key
            for pattern in [f"{cache_key}.*", f"{cache_key}"]:
                for file in self.cache_dir.glob(pattern):
                    file.unlink()


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
