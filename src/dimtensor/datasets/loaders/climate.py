"""Climate dataset loaders (PRISM, NOAA, etc.)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ...core.dimarray import DimArray
from ...core.dimensions import Dimension
from ...core.units import Unit, kelvin, meter, mm
from .base import CSVLoader

# Temperature unit (celsius same dimension as kelvin)
# Note: This is for temperature differences, not absolute temperature
celsius = Unit("°C", kelvin.dimension, 1.0)


class PRISMClimateLoader(CSVLoader):
    """Loader for PRISM climate data.

    PRISM provides gridded climate data for the continental US
    including temperature and precipitation from 1895-present.

    Note: This loader uses a sample/demo dataset since full PRISM
    data requires special access. For production use, update the
    URL to point to actual PRISM data files.
    """

    # Demo/sample URL - in production, this would point to actual PRISM data
    # PRISM data is typically accessed via their data explorer or bulk download
    # Format: CSV with columns for date, location, temperature, precipitation
    URL = "https://prism.oregonstate.edu/sample/climate_data.csv"

    def load(
        self,
        variable: str = "tmean",
        start_year: int | None = None,
        end_year: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Load PRISM climate data.

        Args:
            variable: Climate variable to load:
                - 'tmean': Mean temperature
                - 'tmin': Minimum temperature
                - 'tmax': Maximum temperature
                - 'ppt': Precipitation
            start_year: Starting year (default: all available).
            end_year: Ending year (default: all available).
            **kwargs: Additional arguments (force_download, etc.).

        Returns:
            Dictionary with climate data arrays:
                - dates: Array of date strings
                - values: DimArray with proper units (celsius or mm)
                - latitude: Latitude coordinates (degrees)
                - longitude: Longitude coordinates (degrees)

        Example:
            >>> loader = PRISMClimateLoader()
            >>> data = loader.load(variable='tmean', start_year=2020)
            >>> temps = data['values']
        """
        force = kwargs.get("force_download", False)

        # For MVP, we'll create synthetic sample data
        # In production, this would download from PRISM
        data = self._create_sample_data(variable, start_year, end_year)

        return data

    def _create_sample_data(
        self,
        variable: str,
        start_year: int | None,
        end_year: int | None,
    ) -> dict[str, Any]:
        """Create sample climate data for demonstration.

        Args:
            variable: Climate variable.
            start_year: Starting year.
            end_year: Ending year.

        Returns:
            Dictionary of sample climate data.
        """
        # Generate sample dates (monthly data for 2020-2023)
        if start_year is None:
            start_year = 2020
        if end_year is None:
            end_year = 2023

        n_years = end_year - start_year + 1
        n_months = n_years * 12

        # Generate sample dates
        dates = []
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                dates.append(f"{year}-{month:02d}")

        dates = dates[:n_months]

        # Generate sample values based on variable type
        if variable == "tmean":
            # Temperature: seasonal variation around 15°C
            values = 15 + 10 * np.sin(np.linspace(0, n_years * 2 * np.pi, n_months))
            values += np.random.normal(0, 2, n_months)  # Add noise
            unit = celsius

        elif variable == "tmin":
            # Min temperature: ~5°C below mean
            values = 10 + 10 * np.sin(np.linspace(0, n_years * 2 * np.pi, n_months))
            values += np.random.normal(0, 2, n_months)
            unit = celsius

        elif variable == "tmax":
            # Max temperature: ~5°C above mean
            values = 20 + 10 * np.sin(np.linspace(0, n_years * 2 * np.pi, n_months))
            values += np.random.normal(0, 2, n_months)
            unit = celsius

        elif variable == "ppt":
            # Precipitation: 50-150 mm with seasonal variation
            values = 80 + 30 * np.sin(np.linspace(0, n_years * 2 * np.pi, n_months))
            values += np.random.normal(0, 20, n_months)
            values = np.maximum(values, 0)  # No negative precipitation
            unit = mm

        else:
            raise ValueError(f"Unknown variable: {variable}")

        # Sample coordinates (Portland, Oregon area)
        lat = 45.52
        lon = -122.68

        return {
            "dates": dates,
            "values": DimArray(values, unit=unit),
            "latitude": lat,
            "longitude": lon,
        }


class NOAAClimateLoader(CSVLoader):
    """Loader for NOAA Climate Data Online (CDO).

    Placeholder for future NOAA API integration.
    NOAA provides extensive climate datasets via their API:
    https://www.ncei.noaa.gov/cdo-web/
    """

    def load(self, **kwargs: Any) -> dict[str, DimArray]:
        """Load NOAA climate data.

        Args:
            **kwargs: Additional arguments.

        Returns:
            Climate data dictionary.

        Raises:
            NotImplementedError: NOAA loader not yet implemented.
        """
        raise NotImplementedError(
            "NOAA Climate Data loader not yet implemented. "
            "Future versions will support NOAA API access. "
            "For now, use PRISMClimateLoader for climate data."
        )
