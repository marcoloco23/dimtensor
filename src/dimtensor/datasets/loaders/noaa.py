"""NOAA weather dataset loader.

NOAA Climate Data Online (CDO) API v2 loader for historical weather station data.

API Documentation: https://www.ncdc.noaa.gov/cdo-web/webservices/v2
Token Request: https://www.ncdc.noaa.gov/cdo-web/token

WARNING: The CDO v2 API is deprecated but still functional. Future versions may
migrate to NCEI Data Service API v1: https://www.ncei.noaa.gov/access/services/data/v1

Rate Limits: 5 requests/second, 10,000 requests/day per token.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from ...core.dimarray import DimArray
from ...core.dimensions import Dimension
from ...core.units import Unit, celsius, meter, mm
from .base import BaseLoader

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# Define velocity unit (m/s)
m_per_s = Unit("m/s", Dimension(length=1, time=-1), 1.0)

# Define hectopascal (hPa) - common in meteorology
hectopascal = Unit("hPa", Dimension(mass=1, length=-1, time=-2), 100.0)
hPa = hectopascal


class NOAAWeatherLoader(BaseLoader):
    """Loader for NOAA Climate Data Online (CDO) weather station data.

    Fetches historical weather data from NOAA's CDO v2 API including:
    - Temperature (max, min, average) in celsius
    - Precipitation in mm
    - Wind speed in m/s
    - Atmospheric pressure in hPa

    API token required for real data access. Get token at:
    https://www.ncdc.noaa.gov/cdo-web/token

    Note: CDO v2 API is deprecated but still functional. This loader will
    continue to work but may need migration to v1 API in the future.

    Example:
        >>> # With API token
        >>> loader = NOAAWeatherLoader(token="your_token_here")
        >>> data = loader.load(
        ...     station_id="GHCND:USW00094728",  # Central Park, NYC
        ...     start_date="2023-01-01",
        ...     end_date="2023-12-31",
        ...     variables=["TMAX", "TMIN", "PRCP"]
        ... )
        >>> temps = data['temperature_max']  # DimArray with celsius units

        >>> # Without token (uses sample data)
        >>> loader = NOAAWeatherLoader()
        >>> sample = loader.load()  # Returns synthetic sample data
    """

    API_URL = "https://www.ncdc.noaa.gov/cdo-web/api/v2"
    DATASET_ID = "GHCND"  # Global Historical Climatology Network - Daily

    def __init__(self, token: str | None = None, cache: bool = True):
        """Initialize NOAA weather loader.

        Args:
            token: NOAA API token. If not provided, checks NOAA_API_TOKEN
                environment variable. Without token, sample data is used.
            cache: Whether to enable caching (default: True).
        """
        super().__init__(cache=cache)

        # Get token from parameter or environment
        self.token = token or os.environ.get("NOAA_API_TOKEN")

        # Rate limiting state
        self._last_request_time = 0.0
        self._min_request_interval = 0.2  # 5 requests/second = 0.2s interval

    def load(
        self,
        station_id: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        variables: list[str] | None = None,
        location: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Load NOAA weather station data.

        Args:
            station_id: NOAA station ID (e.g., "GHCND:USW00094728" for NYC Central Park).
                If not provided, will search for nearest station to location.
            start_date: Start date in ISO format (YYYY-MM-DD).
            end_date: End date in ISO format (YYYY-MM-DD).
            variables: List of weather variables to fetch:
                - 'TMAX': Maximum temperature (°C)
                - 'TMIN': Minimum temperature (°C)
                - 'TAVG': Average temperature (°C)
                - 'PRCP': Precipitation (mm)
                - 'AWND': Average wind speed (m/s)
                Default: ['TMAX', 'TMIN', 'PRCP']
            location: (latitude, longitude) tuple for station search if station_id not provided.
            **kwargs: Additional arguments (e.g., max_distance for station search).

        Returns:
            Dictionary containing:
                - station_id: Station identifier
                - station_name: Station name
                - latitude: Station latitude
                - longitude: Station longitude
                - elevation: Station elevation (DimArray with meter units)
                - dates: List of date strings (ISO format)
                - temperature_max: DimArray (celsius) if TMAX requested
                - temperature_min: DimArray (celsius) if TMIN requested
                - temperature_avg: DimArray (celsius) if TAVG requested
                - precipitation: DimArray (mm) if PRCP requested
                - wind_speed: DimArray (m/s) if AWND requested

        Raises:
            ImportError: If requests library not available.
            RuntimeError: If API request fails.

        Example:
            >>> loader = NOAAWeatherLoader(token="your_token")
            >>> data = loader.load(
            ...     station_id="GHCND:USW00094728",
            ...     start_date="2023-01-01",
            ...     end_date="2023-01-31"
            ... )
        """
        # Use sample data if no token provided
        if not self.token:
            return self._create_sample_weather_data(start_date, end_date, variables)

        if not HAS_REQUESTS:
            raise ImportError(
                "requests library required for NOAA API access. "
                "Install with: pip install requests"
            )

        # Default parameters
        if variables is None:
            variables = ["TMAX", "TMIN", "PRCP"]
        if start_date is None:
            start_date = "2023-01-01"
        if end_date is None:
            end_date = "2023-12-31"

        # Find station if not provided
        if station_id is None:
            if location is None:
                # Default to Central Park, NYC
                location = (40.7829, -73.9654)
            station_id = self._search_station(location, start_date, end_date, **kwargs)

        # Fetch weather data
        weather_data = self._fetch_weather_data(
            station_id, start_date, end_date, variables
        )

        # Convert to DimArrays
        return self._convert_to_dimarrays(weather_data, station_id)

    def _search_station(
        self,
        location: tuple[float, float],
        start_date: str,
        end_date: str,
        max_distance: float = 50.0,  # km
        **kwargs: Any,
    ) -> str:
        """Search for weather station near location.

        Args:
            location: (latitude, longitude) tuple.
            start_date: Start date for data availability check.
            end_date: End date for data availability check.
            max_distance: Maximum distance in km (default: 50).
            **kwargs: Additional search parameters.

        Returns:
            Station ID string.

        Raises:
            RuntimeError: If no suitable station found.
        """
        lat, lon = location

        # Search for stations within radius
        # Note: CDO v2 uses extent parameter (west, south, east, north)
        extent = f"{lon-0.5},{lat-0.5},{lon+0.5},{lat+0.5}"

        params = {
            "datasetid": self.DATASET_ID,
            "extent": extent,
            "startdate": start_date,
            "enddate": end_date,
            "limit": 10,  # Get top 10 stations
        }

        url = f"{self.API_URL}/stations"
        response = self._api_request(url, params)

        if "results" not in response or not response["results"]:
            raise RuntimeError(
                f"No NOAA weather stations found near ({lat}, {lon}). "
                f"Try increasing max_distance or providing a specific station_id."
            )

        # Return first station (typically closest)
        station = response["results"][0]
        return station["id"]

    def _fetch_weather_data(
        self,
        station_id: str,
        start_date: str,
        end_date: str,
        variables: list[str],
    ) -> dict[str, Any]:
        """Fetch weather data from NOAA API.

        Args:
            station_id: NOAA station identifier.
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            variables: List of variable codes (e.g., ['TMAX', 'TMIN']).

        Returns:
            Dictionary with raw API response data.

        Raises:
            RuntimeError: If API request fails.
        """
        # Generate cache key (non-security MD5).
        cache_key = hashlib.md5(
            f"{station_id}_{start_date}_{end_date}_{'_'.join(sorted(variables))}".encode(),
            usedforsecurity=False,
        ).hexdigest()

        cache_file = self.cache_dir / f"{cache_key}.json"

        # Check cache
        if self.cache_enabled and cache_file.exists():
            return json.loads(cache_file.read_text())

        # Fetch from API with pagination
        all_results = []
        offset = 1
        limit = 1000  # Max per request

        while True:
            params = {
                "datasetid": self.DATASET_ID,
                "stationid": station_id,
                "startdate": start_date,
                "enddate": end_date,
                "datatypeid": ",".join(variables),
                "limit": limit,
                "offset": offset,
                "units": "metric",
            }

            url = f"{self.API_URL}/data"
            response = self._api_request(url, params)

            if "results" in response:
                all_results.extend(response["results"])

            # Check if more pages
            metadata = response.get("metadata", {})
            result_set = metadata.get("resultset", {})
            count = result_set.get("count", 0)

            if offset + limit > count:
                break

            offset += limit

        # Get station metadata
        station_url = f"{self.API_URL}/stations/{station_id}"
        station_info = self._api_request(station_url)

        # Combine data
        weather_data = {
            "station": station_info,
            "data": all_results,
        }

        # Cache the results
        if self.cache_enabled:
            cache_file.write_text(json.dumps(weather_data, indent=2))

        return weather_data

    def _api_request(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Make rate-limited API request.

        Args:
            url: API endpoint URL.
            params: Query parameters.

        Returns:
            JSON response as dictionary.

        Raises:
            RuntimeError: If request fails.
        """
        # Rate limiting
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)

        headers = {"token": self.token}

        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)
            self._last_request_time = time.time()

            if response.status_code == 429:
                raise RuntimeError(
                    "NOAA API rate limit exceeded. Try again later or reduce request frequency."
                )
            elif response.status_code == 401:
                raise RuntimeError(
                    "Invalid NOAA API token. Get a token at: "
                    "https://www.ncdc.noaa.gov/cdo-web/token"
                )

            response.raise_for_status()
            return response.json()

        except requests.RequestException as e:
            raise RuntimeError(f"NOAA API request failed: {e}") from e

    def _convert_to_dimarrays(
        self,
        weather_data: dict[str, Any],
        station_id: str,
    ) -> dict[str, Any]:
        """Convert raw API response to DimArrays.

        Args:
            weather_data: Raw API response.
            station_id: Station identifier.

        Returns:
            Dictionary with DimArray values.
        """
        station = weather_data["station"]
        data_points = weather_data["data"]

        # Organize data by variable and date
        data_by_var: dict[str, dict[str, float]] = {}
        all_dates = set()

        for point in data_points:
            datatype = point["datatype"]
            date = point["date"][:10]  # Extract YYYY-MM-DD
            value = point["value"]

            if datatype not in data_by_var:
                data_by_var[datatype] = {}

            data_by_var[datatype][date] = value
            all_dates.add(date)

        # Sort dates
        sorted_dates = sorted(all_dates)

        # Create result dictionary
        result: dict[str, Any] = {
            "station_id": station_id,
            "station_name": station.get("name", "Unknown"),
            "latitude": station.get("latitude", 0.0),
            "longitude": station.get("longitude", 0.0),
            "dates": sorted_dates,
        }

        # Add elevation if available
        if "elevation" in station:
            result["elevation"] = DimArray([station["elevation"]], unit=meter)

        # Convert each variable to DimArray
        var_mapping = {
            "TMAX": ("temperature_max", celsius, 10.0),
            "TMIN": ("temperature_min", celsius, 10.0),
            "TAVG": ("temperature_avg", celsius, 10.0),
            "PRCP": ("precipitation", mm, 10.0),
            "AWND": ("wind_speed", m_per_s, 10.0),
            "PRES": ("pressure", hectopascal, 1.0),
        }

        for var_code, (key_name, unit, scale_factor) in var_mapping.items():
            if var_code in data_by_var:
                # Build array with NaN for missing dates
                values = []
                for date in sorted_dates:
                    if date in data_by_var[var_code]:
                        values.append(data_by_var[var_code][date] / scale_factor)
                    else:
                        values.append(np.nan)

                result[key_name] = DimArray(np.array(values), unit=unit)

        return result

    def _create_sample_weather_data(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        variables: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create synthetic sample weather data for demonstration.

        Args:
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            variables: List of variables to include.

        Returns:
            Dictionary with sample weather data.
        """
        if start_date is None:
            start_date = "2023-01-01"
        if end_date is None:
            end_date = "2023-12-31"
        if variables is None:
            variables = ["TMAX", "TMIN", "PRCP"]

        # Parse dates
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
        n_days = (end - start).days + 1

        # Generate date strings using timedelta
        dates = []
        for i in range(n_days):
            current = start + timedelta(days=i)
            dates.append(current.strftime("%Y-%m-%d"))

        # Generate synthetic data with seasonal patterns
        day_of_year = np.arange(n_days)
        seasonal_phase = 2 * np.pi * day_of_year / 365.25

        result: dict[str, Any] = {
            "station_id": "SAMPLE:STATION",
            "station_name": "Sample Weather Station",
            "latitude": 40.7829,
            "longitude": -73.9654,
            "elevation": DimArray([50.0], unit=meter),
            "dates": dates,
        }

        # Generate requested variables
        if "TMAX" in variables:
            tmax = 15 + 15 * np.sin(seasonal_phase) + np.random.normal(0, 3, n_days)
            result["temperature_max"] = DimArray(tmax, unit=celsius)

        if "TMIN" in variables:
            tmin = 5 + 15 * np.sin(seasonal_phase) + np.random.normal(0, 3, n_days)
            result["temperature_min"] = DimArray(tmin, unit=celsius)

        if "TAVG" in variables:
            tavg = 10 + 15 * np.sin(seasonal_phase) + np.random.normal(0, 2, n_days)
            result["temperature_avg"] = DimArray(tavg, unit=celsius)

        if "PRCP" in variables:
            # Precipitation: slightly more in summer
            prcp = 2 + np.sin(seasonal_phase) + np.abs(np.random.normal(0, 3, n_days))
            result["precipitation"] = DimArray(prcp, unit=mm)

        if "AWND" in variables:
            # Wind speed: slightly higher in winter
            wind = 4 + 2 * np.sin(seasonal_phase + np.pi) + np.abs(np.random.normal(0, 1, n_days))
            result["wind_speed"] = DimArray(wind, unit=m_per_s)

        return result
