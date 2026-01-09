"""Astronomy dataset loaders (NASA Exoplanet Archive, etc.)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ...core.dimarray import DimArray
from ...core.dimensions import Dimension
from ...core.units import Unit, day, kg, meter
from .base import CSVLoader

# Astronomical units (not in core units.py yet)
earth_mass = kg * 5.972e24  # Earth mass in kg
earth_radius = meter * 6.371e6  # Earth radius in meters
jupiter_mass = kg * 1.898e27  # Jupiter mass in kg
jupiter_radius = meter * 6.9911e7  # Jupiter radius in meters


class NASAExoplanetLoader(CSVLoader):
    """Loader for NASA Exoplanet Archive confirmed planets.

    Downloads CSV data from NASA Exoplanet Archive with confirmed
    exoplanet discoveries including mass, radius, orbital period, etc.
    """

    # NASA Exoplanet Archive TAP service - Planetary Systems composite table
    # This is the simplified CSV export for confirmed planets
    URL = (
        "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
        "query=select+pl_name,pl_masse,pl_rade,pl_orbper,pl_orbsmax,"
        "st_mass,st_rad+from+ps+where+default_flag=1&format=csv"
    )

    def load(
        self,
        confirmed_only: bool = True,
        columns: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Load NASA Exoplanet Archive data.

        Args:
            confirmed_only: Only load confirmed exoplanets (default: True).
            columns: List of columns to include (None = all).
            **kwargs: Additional arguments (force_download, etc.).

        Returns:
            Dictionary of arrays with exoplanet properties:
                - pl_name: Planet names (list of strings)
                - pl_masse: Planet mass (Earth masses)
                - pl_rade: Planet radius (Earth radii)
                - pl_orbper: Orbital period (days)
                - pl_orbsmax: Semi-major axis (AU)
                - st_mass: Stellar mass (Solar masses)
                - st_rad: Stellar radius (Solar radii)

        Example:
            >>> loader = NASAExoplanetLoader()
            >>> data = loader.load()
            >>> masses = data['pl_masse']
            >>> print(f"Found {len(masses)} exoplanets")
        """
        force = kwargs.get("force_download", False)

        # Download the CSV file
        filepath = self.download(
            self.URL,
            cache_key="nasa_exoplanets_confirmed",
            force=force,
        )

        # Parse the CSV
        data = self._parse_exoplanet_csv(filepath, columns=columns)

        return data

    def _parse_exoplanet_csv(
        self,
        filepath: Path,
        columns: list[str] | None = None,
    ) -> dict[str, Any]:
        """Parse NASA Exoplanet CSV file.

        Args:
            filepath: Path to downloaded CSV.
            columns: Columns to include (None = all).

        Returns:
            Dictionary of DimArray values with proper units.
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

                # Handle missing values
                if not value_str or value_str.lower() in ["", "null", "nan"]:
                    data_lists[col_name].append(np.nan)
                else:
                    # Try to parse as float for numeric columns
                    if col_name != "pl_name":
                        try:
                            data_lists[col_name].append(float(value_str))
                        except ValueError:
                            data_lists[col_name].append(np.nan)
                    else:
                        data_lists[col_name].append(value_str)

        # Convert to DimArrays with proper units
        result: dict[str, Any] = {}

        # Planet name (no unit)
        if "pl_name" in data_lists:
            result["pl_name"] = data_lists["pl_name"]

        # Planet mass (Earth masses)
        if "pl_masse" in data_lists and data_lists["pl_masse"]:
            result["pl_masse"] = DimArray(
                np.array(data_lists["pl_masse"]),
                unit=earth_mass,
            )

        # Planet radius (Earth radii)
        if "pl_rade" in data_lists and data_lists["pl_rade"]:
            result["pl_rade"] = DimArray(
                np.array(data_lists["pl_rade"]),
                unit=earth_radius,
            )

        # Orbital period (days)
        if "pl_orbper" in data_lists and data_lists["pl_orbper"]:
            result["pl_orbper"] = DimArray(
                np.array(data_lists["pl_orbper"]),
                unit=day,
            )

        # Semi-major axis (AU)
        if "pl_orbsmax" in data_lists and data_lists["pl_orbsmax"]:
            # AU = 149597870700 meters
            au = meter * 149597870700
            result["pl_orbsmax"] = DimArray(
                np.array(data_lists["pl_orbsmax"]),
                unit=au,
            )

        # Stellar mass (Solar masses)
        if "st_mass" in data_lists and data_lists["st_mass"]:
            # Solar mass ≈ 1.989e30 kg
            solar_mass = kg * 1.989e30
            result["st_mass"] = DimArray(
                np.array(data_lists["st_mass"]),
                unit=solar_mass,
            )

        # Stellar radius (Solar radii)
        if "st_rad" in data_lists and data_lists["st_rad"]:
            # Solar radius ≈ 6.96e8 m
            solar_radius = meter * 6.96e8
            result["st_rad"] = DimArray(
                np.array(data_lists["st_rad"]),
                unit=solar_radius,
            )

        # Filter by requested columns
        if columns is not None:
            result = {k: v for k, v in result.items() if k in columns}

        return result
