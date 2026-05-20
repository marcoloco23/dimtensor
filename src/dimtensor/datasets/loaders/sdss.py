"""Sloan Digital Sky Survey (SDSS) dataset loader.

This module provides a loader for querying the SDSS SkyServer database
and retrieving astronomical data with proper dimensional units.

The loader supports both high-level query builder methods for common
use cases and raw SQL execution for advanced users.

Example:
    >>> from dimtensor.datasets.loaders import SDSSLoader
    >>> loader = SDSSLoader()
    >>>
    >>> # Cone search around M51 (RA=202.47, Dec=47.20)
    >>> galaxies = loader.radial_search(ra=202.47, dec=47.20, radius=0.1)
    >>>
    >>> # Galaxy sample with magnitude and redshift cuts
    >>> sample = loader.get_galaxies(
    ...     redshift_range=(0.05, 0.15),
    ...     magnitude_range=(14.0, 17.5, 'r'),
    ...     limit=1000
    ... )

Reference:
    Abdurro'uf et al., "The Seventeenth Data Release of the Sloan Digital Sky Surveys"
    ApJS 259 35 (2022). DOI: 10.3847/1538-4365/ac4414
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import numpy as np

from ...core.dimarray import DimArray
from ...core.dimensions import DIMENSIONLESS
from ...core.units import Unit, meter, second
from ...domains.astronomy import (
    arcsecond,
    megaparsec,
    parsec,
    solar_mass,
)
from .base import CSVLoader

try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# SDSS-specific units
# Magnitudes are dimensionless (logarithmic scale)
magnitude = Unit("mag", DIMENSIONLESS, 1.0)

# Velocity unit (km/s)
km_per_s = Unit("km/s", meter.dimension / second.dimension, 1000.0)


class SDSSLoader(CSVLoader):
    """Loader for Sloan Digital Sky Survey data via SkyServer API.

    Provides access to SDSS photometric and spectroscopic data through
    both high-level query builder methods and raw SQL execution.

    Attributes:
        data_release: SDSS data release to query (default: 17).
        base_url: SkyServer API base URL.
        timeout: Query timeout in seconds (default: 60).
        max_retries: Maximum number of retry attempts (default: 3).

    Example:
        >>> loader = SDSSLoader(data_release=17)
        >>> # Get galaxies in a cone
        >>> data = loader.radial_search(ra=180.0, dec=0.0, radius=0.5)
        >>> print(f"Found {len(data['ra'])} galaxies")
    """

    def __init__(
        self,
        data_release: int = 17,
        cache: bool = True,
        timeout: int = 60,
        max_retries: int = 3,
    ):
        """Initialize SDSS loader.

        Args:
            data_release: SDSS data release number (default: 17).
            cache: Enable caching of query results (default: True).
            timeout: Query timeout in seconds (default: 60).
            max_retries: Maximum retry attempts on failure (default: 3).
        """
        super().__init__(cache=cache)
        self.data_release = data_release
        self.timeout = timeout
        self.max_retries = max_retries

        # Construct base URL for SkyServer
        self.base_url = (
            f"http://skyserver.sdss.org/dr{data_release}/SkyServerWS/SearchTools/SqlSearch"
        )

    def load(self, **kwargs: Any) -> dict[str, Any]:
        """Load SDSS data (use specific query methods instead).

        This method is not directly used. Use query methods like
        get_galaxies(), radial_search(), or execute_query() instead.

        Raises:
            NotImplementedError: Always raised. Use specific query methods.
        """
        raise NotImplementedError(
            "SDSSLoader requires a specific query method. "
            "Use get_galaxies(), radial_search(), get_spectroscopy(), "
            "or execute_query() instead."
        )

    def execute_query(
        self,
        sql: str,
        limit: int | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Execute raw SQL query against SDSS database.

        Args:
            sql: SQL query string (without LIMIT clause if limit is set).
            limit: Maximum number of rows to return (default: None).
            force: Force re-query even if cached (default: False).

        Returns:
            Dictionary of column names to DimArray values with proper units.

        Example:
            >>> sql = "SELECT ra, dec, z FROM SpecObj WHERE z > 0.5 AND z < 0.6"
            >>> data = loader.execute_query(sql, limit=100)
            >>> redshifts = data['z']

        Raises:
            RuntimeError: If query fails or returns error.
        """
        if not HAS_REQUESTS:
            raise ImportError(
                "requests library required for SDSS queries. "
                "Install with: pip install requests"
            )

        # Add TOP clause if limit specified
        if limit is not None:
            # Check if query already has TOP
            sql_upper = sql.upper()
            if "SELECT" in sql_upper and "TOP" not in sql_upper:
                sql = sql.replace("SELECT", f"SELECT TOP {limit}", 1)
                sql = sql.replace("select", f"SELECT TOP {limit}", 1)

        # Generate cache key from SQL (non-security MD5).
        cache_key = hashlib.md5(sql.encode(), usedforsecurity=False).hexdigest()[:16]
        cache_key = f"sdss_dr{self.data_release}_{cache_key}"

        cache_file = self.cache_dir / f"{cache_key}.csv"

        # Check cache
        if self.cache_enabled and not force and cache_file.exists():
            return self._parse_sdss_csv(cache_file)

        # Build URL
        params = {"cmd": sql, "format": "csv"}
        url = f"{self.base_url}?{urlencode(params)}"

        # Execute query with retries
        response_text = None
        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, timeout=self.timeout)
                response.raise_for_status()
                response_text = response.text
                break
            except requests.RequestException as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    time.sleep(2 ** attempt)
                continue

        if response_text is None:
            raise RuntimeError(
                f"SDSS query failed after {self.max_retries} attempts: {last_error}"
            )

        # Check for error messages in response
        if "error" in response_text.lower()[:200]:
            raise RuntimeError(f"SDSS query error: {response_text[:500]}")

        # Write to cache
        cache_file.write_text(response_text)

        # Parse result
        return self._parse_sdss_csv(cache_file)

    def radial_search(
        self,
        ra: float,
        dec: float,
        radius: float,
        limit: int = 1000,
        spectroscopy: bool = False,
        force: bool = False,
    ) -> dict[str, Any]:
        """Perform cone search around given coordinates.

        Args:
            ra: Right ascension in degrees (J2000).
            dec: Declination in degrees (J2000).
            radius: Search radius in degrees.
            limit: Maximum number of objects to return (default: 1000).
            spectroscopy: Include only objects with spectroscopy (default: False).
            force: Force re-query even if cached (default: False).

        Returns:
            Dictionary with columns: ra, dec, objID, u, g, r, i, z magnitudes,
            and redshift (z) if spectroscopy=True.

        Example:
            >>> # Search around M51
            >>> data = loader.radial_search(ra=202.47, dec=47.20, radius=0.1)
            >>> print(f"Found {len(data['ra'])} objects")
        """
        if spectroscopy:
            # Use SpecObj table with fGetNearbySpecObjEq function
            sql = f"""
            SELECT
                s.ra, s.dec, s.objID, s.z, s.zErr,
                s.plate, s.mjd, s.fiberID,
                p.u, p.g, p.r, p.i, p.z as z_mag
            FROM fGetNearbySpecObjEq({ra}, {dec}, {radius}) AS nearby
            JOIN SpecObj s ON nearby.objID = s.objID
            JOIN PhotoObj p ON s.objID = p.objID
            WHERE s.sciencePrimary = 1 AND s.zWarning = 0
            """
        else:
            # Use PhotoObj table with fGetNearbyObjEq function
            sql = f"""
            SELECT
                p.ra, p.dec, p.objID,
                p.u, p.g, p.r, p.i, p.z as z_mag,
                p.petroRad_r, p.petroR50_r,
                p.type
            FROM fGetNearbyObjEq({ra}, {dec}, {radius}) AS nearby
            JOIN PhotoObj p ON nearby.objID = p.objID
            WHERE p.mode = 1 AND p.clean = 1
            """

        return self.execute_query(sql, limit=limit, force=force)

    def get_galaxies(
        self,
        ra_range: tuple[float, float] | None = None,
        dec_range: tuple[float, float] | None = None,
        redshift_range: tuple[float, float] | None = None,
        magnitude_range: tuple[float, float, str] | None = None,
        limit: int = 1000,
        force: bool = False,
    ) -> dict[str, Any]:
        """Get galaxy sample with optional filters.

        Args:
            ra_range: (min_ra, max_ra) in degrees (default: None).
            dec_range: (min_dec, max_dec) in degrees (default: None).
            redshift_range: (z_min, z_max) for spectroscopic redshift (default: None).
            magnitude_range: (mag_min, mag_max, band) where band is u/g/r/i/z (default: None).
            limit: Maximum number of galaxies to return (default: 1000).
            force: Force re-query even if cached (default: False).

        Returns:
            Dictionary with galaxy properties as DimArrays.

        Example:
            >>> # Get bright galaxies at intermediate redshift
            >>> galaxies = loader.get_galaxies(
            ...     redshift_range=(0.05, 0.1),
            ...     magnitude_range=(14.0, 17.0, 'r'),
            ...     limit=500
            ... )
        """
        # Build WHERE clauses
        conditions = []

        # Galaxy classification (type = 3 in PhotoObj)
        conditions.append("p.type = 3")

        # Quality flags
        conditions.append("p.mode = 1")  # Primary observation
        conditions.append("p.clean = 1")  # Clean photometry

        # Coordinate ranges
        if ra_range is not None:
            conditions.append(f"p.ra BETWEEN {ra_range[0]} AND {ra_range[1]}")

        if dec_range is not None:
            conditions.append(f"p.dec BETWEEN {dec_range[0]} AND {dec_range[1]}")

        # Magnitude range
        if magnitude_range is not None:
            mag_min, mag_max, band = magnitude_range
            if band not in ['u', 'g', 'r', 'i', 'z']:
                raise ValueError(f"Invalid band: {band}. Must be u, g, r, i, or z.")
            conditions.append(f"p.{band} BETWEEN {mag_min} AND {mag_max}")

        # Build query based on whether we need spectroscopy
        if redshift_range is not None:
            # Join with SpecObj for redshift
            z_min, z_max = redshift_range
            conditions.append(f"s.z BETWEEN {z_min} AND {z_max}")
            conditions.append("s.sciencePrimary = 1")
            conditions.append("s.zWarning = 0")

            sql = f"""
            SELECT
                p.ra, p.dec, p.objID,
                p.u, p.g, p.r, p.i, p.z as z_mag,
                p.petroRad_r, p.petroR50_r,
                s.z, s.zErr, s.plate, s.mjd, s.fiberID
            FROM PhotoObj p
            JOIN SpecObj s ON p.objID = s.objID
            WHERE {' AND '.join(conditions)}
            """
        else:
            # PhotoObj only
            sql = f"""
            SELECT
                p.ra, p.dec, p.objID,
                p.u, p.g, p.r, p.i, p.z as z_mag,
                p.petroRad_r, p.petroR50_r,
                p.type
            FROM PhotoObj p
            WHERE {' AND '.join(conditions)}
            """

        return self.execute_query(sql, limit=limit, force=force)

    def get_spectroscopy(
        self,
        objIDs: list[int] | None = None,
        plate_mjd_fiber: list[tuple[int, int, int]] | None = None,
        limit: int = 1000,
        force: bool = False,
    ) -> dict[str, Any]:
        """Get spectroscopic data for specific objects.

        Args:
            objIDs: List of SDSS object IDs (default: None).
            plate_mjd_fiber: List of (plate, mjd, fiber) tuples (default: None).
            limit: Maximum number of spectra to return (default: 1000).
            force: Force re-query even if cached (default: False).

        Returns:
            Dictionary with spectroscopic properties as DimArrays.

        Example:
            >>> # Get spectra for specific objects
            >>> data = loader.get_spectroscopy(objIDs=[1237648720693755918])
            >>> redshift = data['z']

        Raises:
            ValueError: If neither objIDs nor plate_mjd_fiber is provided.
        """
        if objIDs is None and plate_mjd_fiber is None:
            raise ValueError("Must provide either objIDs or plate_mjd_fiber")

        # Build WHERE clause
        if objIDs is not None:
            # Use objID matching
            id_list = ",".join(str(oid) for oid in objIDs)
            where_clause = f"s.objID IN ({id_list})"
        else:
            # Use plate/mjd/fiber matching
            conditions = []
            for plate, mjd, fiber in plate_mjd_fiber:  # type: ignore
                conditions.append(
                    f"(s.plate = {plate} AND s.mjd = {mjd} AND s.fiberID = {fiber})"
                )
            where_clause = " OR ".join(conditions)

        sql = f"""
        SELECT
            s.objID, s.ra, s.dec,
            s.z, s.zErr, s.zWarning,
            s.plate, s.mjd, s.fiberID,
            s.class, s.subClass,
            p.u, p.g, p.r, p.i, p.z as z_mag
        FROM SpecObj s
        JOIN PhotoObj p ON s.objID = p.objID
        WHERE ({where_clause}) AND s.sciencePrimary = 1
        """

        return self.execute_query(sql, limit=limit, force=force)

    def _parse_sdss_csv(self, filepath: Path) -> dict[str, Any]:
        """Parse SDSS CSV result into DimArrays with proper units.

        Args:
            filepath: Path to CSV file from SDSS query.

        Returns:
            Dictionary mapping column names to DimArray values.
        """
        rows = self.parse_csv(filepath, skip_rows=1)  # Skip header

        if not rows:
            return {}

        # Parse header
        header_line = filepath.read_text().split("\n")[0]
        header = [col.strip() for col in header_line.split(",")]

        # Build column index map
        col_map = {name: i for i, name in enumerate(header)}

        # Initialize data lists
        data_lists: dict[str, list[Any]] = {name: [] for name in header}

        # Parse each row
        for row in rows:
            if len(row) != len(header):
                continue  # Skip malformed rows

            for col_name, col_idx in col_map.items():
                value_str = row[col_idx]

                # Handle missing values (SDSS uses -9999 and empty strings)
                if not value_str or value_str in ["", "null", "NULL"]:
                    data_lists[col_name].append(np.nan)
                elif value_str in ["-9999", "-9999.0"]:
                    data_lists[col_name].append(np.nan)
                else:
                    # Try to parse as number
                    try:
                        # Try int first for objID, plate, mjd, fiberID
                        if col_name in ['objID', 'plate', 'mjd', 'fiberID', 'type']:
                            data_lists[col_name].append(int(float(value_str)))
                        else:
                            data_lists[col_name].append(float(value_str))
                    except ValueError:
                        # Keep as string (for class, subClass columns)
                        data_lists[col_name].append(value_str)

        # Convert to DimArrays with proper units
        result: dict[str, Any] = {}

        # Define unit mappings
        unit_map = {
            # Coordinates (degrees - stored as dimensionless)
            'ra': DIMENSIONLESS,
            'dec': DIMENSIONLESS,
            # Magnitudes (dimensionless, AB system)
            'u': magnitude,
            'g': magnitude,
            'r': magnitude,
            'i': magnitude,
            'z_mag': magnitude,
            # Redshift (dimensionless)
            'z': DIMENSIONLESS,
            'zErr': DIMENSIONLESS,
            # Angular sizes (arcseconds)
            'petroRad_r': arcsecond,
            'petroR50_r': arcsecond,
        }

        for col_name, values in data_lists.items():
            if not values:
                continue

            # Get unit for this column
            unit = unit_map.get(col_name)

            if unit is not None:
                # Convert to numpy array and create DimArray
                arr = np.array(values)
                result[col_name] = DimArray(arr, unit=Unit("", unit, 1.0))
            else:
                # Keep as plain list (objID, plate, type, class, etc.)
                result[col_name] = values

        return result
