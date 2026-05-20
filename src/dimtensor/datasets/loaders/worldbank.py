"""World Bank Climate Data API loader."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from ...core.dimarray import DimArray
from ...core.units import kelvin, mm
from .base import BaseLoader

# Temperature unit (celsius same dimension as kelvin)
# Note: This is for temperature differences, not absolute temperature
celsius = kelvin.__class__("°C", kelvin.dimension, 1.0)


class WorldBankClimateLoader(BaseLoader):
    """Loader for World Bank Climate Data API.

    Provides access to historical and projected climate data from the
    World Bank Climate Data API, including temperature and precipitation
    at country and basin spatial scales.

    Data sources:
    - Historical: CRU (Climate Research Unit) data
    - Projections: 15 Global Circulation Models (GCMs)

    API Documentation:
    https://datahelpdesk.worldbank.org/knowledgebase/articles/902061-climate-data-api

    Example:
        >>> loader = WorldBankClimateLoader()
        >>> # Load historical temperature data for Kenya
        >>> data = loader.load(
        ...     country_code="KEN",
        ...     variable="tas",
        ...     data_type="cru",
        ...     temporal_scale="year"
        ... )
        >>> temps = data['values']  # DimArray with celsius units
        >>> years = data['years']   # Array of years

        >>> # Load precipitation data for USA
        >>> data = loader.load(
        ...     country_code="USA",
        ...     variable="pr",
        ...     data_type="cru",
        ...     temporal_scale="month"
        ... )
        >>> precip = data['values']  # DimArray with mm units
    """

    BASE_URL = "http://climatedataapi.worldbank.org/climateweb/rest/v1"

    # Common GCM models for projections
    GCM_MODELS = [
        "bccr_bcm2_0",
        "csiro_mk3_5",
        "ingv_echam4",
        "cccma_cgcm3_1",
        "cnrm_cm3",
        "gfdl_cm2_0",
        "gfdl_cm2_1",
        "ipsl_cm4",
        "microc3_2_medres",
        "miub_echo_g",
        "mpi_echam5",
        "mri_cgcm2_3_2a",
        "inmcm3_0",
        "ukmo_hadcm3",
        "ukmo_hadgem1",
    ]

    def __init__(self, cache: bool = True, retry_attempts: int = 3):
        """Initialize the World Bank Climate loader.

        Args:
            cache: Whether to enable caching (default: True).
            retry_attempts: Number of retry attempts for API calls (default: 3).
        """
        super().__init__(cache=cache)
        self.retry_attempts = retry_attempts

    def load(
        self,
        country_code: str,
        variable: str = "tas",
        data_type: str = "cru",
        temporal_scale: str = "year",
        start_year: int | None = None,
        end_year: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Load climate data for a country.

        Args:
            country_code: ISO3 country code (e.g., "USA", "KEN", "DEU").
            variable: Climate variable to load:
                - "tas": Temperature (in Celsius)
                - "pr": Precipitation (in mm)
            data_type: Data source:
                - "cru": Historical data (1901-2009 for countries)
                - GCM model name: Projected data (see GCM_MODELS)
            temporal_scale: Temporal resolution:
                - "year": Annual data
                - "month": Monthly data
            start_year: Filter to data >= this year (optional).
            end_year: Filter to data <= this year (optional).
            **kwargs: Additional arguments (force_download, etc.).

        Returns:
            Dictionary with climate data:
                - values: DimArray with proper units (celsius or mm)
                - years: Array of years (for annual data)
                - months: Array of month indices 1-12 (for monthly data)
                - time: Array of time labels (year or year-month)
                - metadata: Dict with query parameters and data info

        Raises:
            ValueError: If invalid parameters provided.
            RuntimeError: If API request fails.

        Example:
            >>> loader = WorldBankClimateLoader()
            >>> data = loader.load("KEN", variable="tas", temporal_scale="year")
            >>> print(f"Temperature: {data['values'].mean()}")
        """
        # Validate inputs
        if variable not in ["tas", "pr"]:
            raise ValueError(
                f"Invalid variable: {variable}. Must be 'tas' or 'pr'."
            )

        if temporal_scale not in ["year", "month"]:
            raise ValueError(
                f"Invalid temporal_scale: {temporal_scale}. "
                "Must be 'year' or 'month'."
            )

        if data_type not in ["cru"] + self.GCM_MODELS:
            raise ValueError(
                f"Invalid data_type: {data_type}. "
                f"Must be 'cru' or one of: {', '.join(self.GCM_MODELS)}"
            )

        # Build API URL
        url = self._build_country_url(
            country_code=country_code,
            variable=variable,
            data_type=data_type,
            temporal_scale=temporal_scale,
        )

        # Fetch data with retry logic
        force = kwargs.get("force_download", False)
        response_data = self._fetch_with_retry(url, force=force)

        # Parse response
        parsed_data = self._parse_climate_response(
            response_data,
            variable=variable,
            temporal_scale=temporal_scale,
            start_year=start_year,
            end_year=end_year,
        )

        # Add metadata
        parsed_data["metadata"] = {
            "country_code": country_code,
            "variable": variable,
            "data_type": data_type,
            "temporal_scale": temporal_scale,
            "api_url": url,
        }

        return parsed_data

    def load_basin(
        self,
        basin_id: int,
        variable: str = "tas",
        data_type: str = "cru",
        temporal_scale: str = "year",
        start_year: int | None = None,
        end_year: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Load climate data for a river basin.

        Args:
            basin_id: Numeric basin identifier.
            variable: Climate variable ("tas" or "pr").
            data_type: Data source ("cru" or GCM model).
            temporal_scale: Temporal resolution ("year" or "month").
            start_year: Filter to data >= this year (optional).
            end_year: Filter to data <= this year (optional).
            **kwargs: Additional arguments (force_download, etc.).

        Returns:
            Dictionary with climate data (same format as load()).

        Example:
            >>> loader = WorldBankClimateLoader()
            >>> data = loader.load_basin(basin_id=302, variable="pr")
        """
        # Validate inputs (same as load())
        if variable not in ["tas", "pr"]:
            raise ValueError(
                f"Invalid variable: {variable}. Must be 'tas' or 'pr'."
            )

        if temporal_scale not in ["year", "month"]:
            raise ValueError(
                f"Invalid temporal_scale: {temporal_scale}. "
                "Must be 'year' or 'month'."
            )

        if data_type not in ["cru"] + self.GCM_MODELS:
            raise ValueError(
                f"Invalid data_type: {data_type}. "
                f"Must be 'cru' or one of: {', '.join(self.GCM_MODELS)}"
            )

        # Build API URL
        url = self._build_basin_url(
            basin_id=basin_id,
            variable=variable,
            data_type=data_type,
            temporal_scale=temporal_scale,
        )

        # Fetch data with retry logic
        force = kwargs.get("force_download", False)
        response_data = self._fetch_with_retry(url, force=force)

        # Parse response
        parsed_data = self._parse_climate_response(
            response_data,
            variable=variable,
            temporal_scale=temporal_scale,
            start_year=start_year,
            end_year=end_year,
        )

        # Add metadata
        parsed_data["metadata"] = {
            "basin_id": basin_id,
            "variable": variable,
            "data_type": data_type,
            "temporal_scale": temporal_scale,
            "api_url": url,
        }

        return parsed_data

    def _build_country_url(
        self,
        country_code: str,
        variable: str,
        data_type: str,
        temporal_scale: str,
    ) -> str:
        """Build API URL for country query.

        Format: /v1/country/{data_type}/{var}/{temporal_scale}/{ISO3}.json
        """
        return (
            f"{self.BASE_URL}/country/{data_type}/{variable}/"
            f"{temporal_scale}/{country_code}.json"
        )

    def _build_basin_url(
        self,
        basin_id: int,
        variable: str,
        data_type: str,
        temporal_scale: str,
    ) -> str:
        """Build API URL for basin query.

        Format: /v1/basin/{data_type}/{var}/{temporal_scale}/{basinID}.json
        """
        return (
            f"{self.BASE_URL}/basin/{data_type}/{variable}/"
            f"{temporal_scale}/{basin_id}.json"
        )

    def _fetch_with_retry(
        self,
        url: str,
        force: bool = False,
    ) -> list[dict[str, Any]]:
        """Fetch data from API with retry logic and exponential backoff.

        Args:
            url: API endpoint URL.
            force: Force re-download even if cached.

        Returns:
            Parsed JSON response (list of data points).

        Raises:
            ImportError: If requests library not available.
            RuntimeError: If all retry attempts fail.
        """
        if not HAS_REQUESTS:
            raise ImportError(
                "requests library required for World Bank Climate loader. "
                "Install with: pip install requests"
            )

        # Check cache first (using URL as cache key; non-security MD5).
        import hashlib
        cache_key = hashlib.md5(url.encode(), usedforsecurity=False).hexdigest()
        cache_file = self.cache_dir / f"{cache_key}.json"

        if self.cache_enabled and not force and cache_file.exists():
            import json
            return json.loads(cache_file.read_text())

        # Retry with exponential backoff
        last_error = None
        for attempt in range(self.retry_attempts):
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()

                # Parse JSON response
                data = response.json()

                # Cache the response
                if self.cache_enabled:
                    import json
                    cache_file.write_text(json.dumps(data, indent=2))

                return data

            except requests.RequestException as e:
                last_error = e
                if attempt < self.retry_attempts - 1:
                    # Exponential backoff: 1s, 2s, 4s
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                continue

        # All retries failed
        raise RuntimeError(
            f"Failed to fetch data from World Bank API after "
            f"{self.retry_attempts} attempts: {last_error}"
        ) from last_error

    def _parse_climate_response(
        self,
        response_data: list[dict[str, Any]],
        variable: str,
        temporal_scale: str,
        start_year: int | None = None,
        end_year: int | None = None,
    ) -> dict[str, Any]:
        """Parse JSON response from World Bank Climate API.

        Args:
            response_data: List of data points from API.
            variable: Climate variable ("tas" or "pr").
            temporal_scale: Temporal resolution ("year" or "month").
            start_year: Filter to data >= this year.
            end_year: Filter to data <= this year.

        Returns:
            Dictionary with parsed data and DimArrays.
        """
        if not response_data:
            raise ValueError("Empty response from API")

        # Parse based on temporal scale
        if temporal_scale == "year":
            years = []
            values = []

            for item in response_data:
                year = item.get("year")
                value = item.get(variable)

                # Filter by year range if specified
                if start_year is not None and year < start_year:
                    continue
                if end_year is not None and year > end_year:
                    continue

                years.append(year)

                # Handle missing data
                if value is None or value == "":
                    values.append(np.nan)
                else:
                    values.append(float(value))

            result = {
                "years": np.array(years),
                "time": [str(y) for y in years],
                "values": self._create_dimarray(values, variable),
            }

        elif temporal_scale == "month":
            months = []
            years = []
            values = []
            time_labels = []

            for item in response_data:
                year = item.get("year")
                month = item.get("month")
                value = item.get(variable)

                # Filter by year range if specified
                if start_year is not None and year < start_year:
                    continue
                if end_year is not None and year > end_year:
                    continue

                years.append(year)
                months.append(month)
                time_labels.append(f"{year}-{month:02d}")

                # Handle missing data
                if value is None or value == "":
                    values.append(np.nan)
                else:
                    values.append(float(value))

            result = {
                "years": np.array(years),
                "months": np.array(months),
                "time": time_labels,
                "values": self._create_dimarray(values, variable),
            }

        else:
            raise ValueError(f"Invalid temporal_scale: {temporal_scale}")

        return result

    def _create_dimarray(
        self,
        values: list[float],
        variable: str,
    ) -> DimArray:
        """Create DimArray with appropriate units.

        Args:
            values: List of numeric values.
            variable: Variable type ("tas" or "pr").

        Returns:
            DimArray with proper units.
        """
        arr = np.array(values)

        if variable == "tas":
            # Temperature in Celsius
            unit = celsius
        elif variable == "pr":
            # Precipitation in millimeters
            unit = mm
        else:
            raise ValueError(f"Unknown variable: {variable}")

        return DimArray(arr, unit=unit)
