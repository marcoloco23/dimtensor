"""Gravitational wave dataset loaders (GWOSC/LIGO/Virgo)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...core.dimarray import DimArray
from ...core.dimensions import DIMENSIONLESS
from ...core.units import Unit, hertz, kg, meter, second
from .base import BaseLoader

# Check for gwosc availability
try:
    import gwosc
    from gwosc import datasets as gwosc_datasets
    HAS_GWOSC = True
except ImportError:
    HAS_GWOSC = False

# Astronomical units for gravitational wave astronomy
solar_mass = kg * 1.989e30  # Solar mass in kg
parsec = meter * 3.0857e16  # Parsec in meters
megaparsec = parsec * 1e6  # Megaparsec

# Strain is dimensionless (fractional length change ΔL/L)
strain = Unit("strain", DIMENSIONLESS, 1.0)


class GWOSCEventLoader(BaseLoader):
    """Loader for gravitational wave event catalogs from GWOSC.

    Downloads event catalogs (GWTC-1, GWTC-2, GWTC-3, GWTC-4) from the
    Gravitational Wave Open Science Center and converts event parameters
    to DimArrays with proper units.

    Requires the gwosc package: pip install gwosc

    Example:
        >>> loader = GWOSCEventLoader()
        >>> events = loader.load(catalog='GWTC-3')
        >>> print(f"Found {len(events['mass_1'])} events")
        >>> print(f"Primary mass: {events['mass_1'][0]}")  # In solar masses
    """

    def __init__(self, cache: bool = True):
        """Initialize the loader.

        Args:
            cache: Whether to enable caching (default: True).

        Raises:
            ImportError: If gwosc package is not installed.
        """
        if not HAS_GWOSC:
            raise ImportError(
                "gwosc package required for gravitational wave data loading. "
                "Install with: pip install gwosc"
            )
        super().__init__(cache=cache)

    def load(
        self,
        catalog: str = "GWTC-3",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Load gravitational wave event catalog.

        Args:
            catalog: Catalog name (e.g., 'GWTC-1', 'GWTC-2', 'GWTC-3', 'GWTC-4').
                    Default: 'GWTC-3'.
            **kwargs: Additional arguments (unused, for compatibility).

        Returns:
            Dictionary of event parameters with proper units:
                - event_names: List of event names (strings)
                - mass_1: Primary mass (solar masses)
                - mass_2: Secondary mass (solar masses)
                - luminosity_distance: Distance (megaparsecs)
                - chirp_mass: Chirp mass (solar masses)
                - final_mass: Final mass (solar masses)
                - GPS: GPS time of event (seconds)
                - network_matched_filter_snr: Signal-to-noise ratio

        Example:
            >>> loader = GWOSCEventLoader()
            >>> events = loader.load(catalog='GWTC-3')
            >>> print(f"First event: {events['event_names'][0]}")
            >>> print(f"Primary mass: {events['mass_1'][0]}")
        """
        # Get catalog data from GWOSC
        try:
            # gwosc.datasets.find_datasets returns event list for catalogs
            event_list = gwosc_datasets.find_datasets(
                type="events",
                catalog=catalog
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load catalog '{catalog}' from GWOSC. "
                f"Available catalogs: GWTC-1, GWTC-2, GWTC-3, GWTC-4. "
                f"Error: {e}"
            ) from e

        if not event_list:
            raise ValueError(f"No events found in catalog '{catalog}'")

        # Initialize data lists
        event_names: list[str] = []
        mass_1_list: list[float] = []
        mass_2_list: list[float] = []
        distance_list: list[float] = []
        chirp_mass_list: list[float] = []
        final_mass_list: list[float] = []
        gps_list: list[float] = []
        snr_list: list[float] = []

        # Fetch event parameters
        for event_name in event_list:
            try:
                # Get event GPS time
                gps_time = gwosc_datasets.event_gps(event_name)

                # Get event parameters (this returns a dictionary)
                # Note: gwosc may not have detailed parameters for all events
                # We'll try to get them, but handle missing gracefully
                try:
                    from gwosc import api
                    event_data = api.fetch_event_json(event_name)
                    params = event_data.get("events", {}).get(event_name, {}).get("parameters", {})
                except Exception:
                    # If we can't get detailed parameters, skip this event
                    continue

                # Extract parameters (with fallback to NaN)
                m1 = params.get("mass_1_source", np.nan)
                m2 = params.get("mass_2_source", np.nan)
                dist = params.get("luminosity_distance", np.nan)
                mc = params.get("chirp_mass_source", np.nan)
                mf = params.get("final_mass_source", np.nan)
                snr = params.get("network_matched_filter_snr", np.nan)

                event_names.append(event_name)
                mass_1_list.append(m1 if m1 is not None else np.nan)
                mass_2_list.append(m2 if m2 is not None else np.nan)
                distance_list.append(dist if dist is not None else np.nan)
                chirp_mass_list.append(mc if mc is not None else np.nan)
                final_mass_list.append(mf if mf is not None else np.nan)
                gps_list.append(gps_time)
                snr_list.append(snr if snr is not None else np.nan)

            except Exception:
                # Skip events that fail to load
                continue

        if not event_names:
            raise RuntimeError(
                f"Failed to load any event parameters from catalog '{catalog}'. "
                "The GWOSC API may have changed or the catalog may not have "
                "detailed parameters available."
            )

        # Convert to DimArrays with proper units
        result: dict[str, Any] = {
            "event_names": event_names,
            "mass_1": DimArray(np.array(mass_1_list), unit=solar_mass),
            "mass_2": DimArray(np.array(mass_2_list), unit=solar_mass),
            "luminosity_distance": DimArray(np.array(distance_list), unit=megaparsec),
            "chirp_mass": DimArray(np.array(chirp_mass_list), unit=solar_mass),
            "final_mass": DimArray(np.array(final_mass_list), unit=solar_mass),
            "GPS": DimArray(np.array(gps_list), unit=second),
            "network_matched_filter_snr": np.array(snr_list),  # Dimensionless, no unit
        }

        return result


class GWOSCStrainLoader(BaseLoader):
    """Loader for gravitational wave strain data from GWOSC.

    Downloads strain time series data h(t) from LIGO/Virgo/GEO detectors
    and converts to DimArrays with proper time and strain units.

    Requires the gwosc package: pip install gwosc

    Example:
        >>> loader = GWOSCStrainLoader()
        >>> data = loader.load(
        ...     event='GW150914',
        ...     detector='H1',
        ...     duration=32,
        ...     sample_rate=4096
        ... )
        >>> print(f"Strain shape: {data['strain'].shape}")
        >>> print(f"Time axis: {data['time']}")
    """

    def __init__(self, cache: bool = True):
        """Initialize the loader.

        Args:
            cache: Whether to enable caching (default: True).

        Raises:
            ImportError: If gwosc package is not installed.
        """
        if not HAS_GWOSC:
            raise ImportError(
                "gwosc package required for gravitational wave data loading. "
                "Install with: pip install gwosc"
            )
        super().__init__(cache=cache)

    def load(
        self,
        event: str | None = None,
        detector: str = "H1",
        duration: float = 32.0,
        sample_rate: int = 4096,
        gps_time: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Load gravitational wave strain data.

        Args:
            event: Event name (e.g., 'GW150914'). If provided, gps_time is
                  automatically determined. Either event or gps_time must be provided.
            detector: Detector name ('H1', 'L1', 'V1', etc.). Default: 'H1'.
            duration: Duration of data to load in seconds, centered on event.
                     Default: 32 seconds.
            sample_rate: Sample rate in Hz (typically 4096 or 16384).
                        Default: 4096 Hz.
            gps_time: GPS time to center data around. If not provided, uses
                     event GPS time.
            **kwargs: Additional arguments (unused).

        Returns:
            Dictionary containing:
                - strain: Strain time series (dimensionless)
                - time: Time axis (seconds relative to GPS time)
                - gps_start: GPS start time (seconds)
                - sample_rate: Sample rate (Hz)
                - detector: Detector name (string)
                - event: Event name (string, if provided)

        Raises:
            ValueError: If neither event nor gps_time is provided, or if
                       data is not available for the specified parameters.

        Example:
            >>> loader = GWOSCStrainLoader()
            >>> data = loader.load(event='GW150914', detector='H1')
            >>> strain = data['strain']
            >>> time = data['time']
        """
        # Determine GPS time
        if event is not None:
            try:
                center_gps = gwosc_datasets.event_gps(event)
            except Exception as e:
                raise ValueError(
                    f"Failed to get GPS time for event '{event}'. "
                    f"Error: {e}"
                ) from e
        elif gps_time is not None:
            center_gps = gps_time
        else:
            raise ValueError("Either 'event' or 'gps_time' must be provided")

        # Calculate start and end GPS times
        half_duration = duration / 2.0
        gps_start = center_gps - half_duration
        gps_end = center_gps + half_duration

        # Warn if downloading large data
        data_size_mb = (duration * sample_rate * 8) / (1024 * 1024)  # rough estimate
        if data_size_mb > 10:
            import warnings
            warnings.warn(
                f"Downloading ~{data_size_mb:.1f} MB of strain data. "
                f"This may take some time.",
                UserWarning,
                stacklevel=2,
            )

        # Fetch strain data using gwosc
        try:
            # Use gwosc.datasets.fetch_open_data which returns strain TimeSeries
            # This automatically handles the download and caching
            urls = gwosc_datasets.fetch_open_data(
                detector,
                gps_start,
                gps_end,
                sample_rate=sample_rate,
                format='numpy'
            )

            if urls is None:
                raise ValueError(
                    f"No data available for detector '{detector}' "
                    f"at GPS time {center_gps} (±{half_duration}s) "
                    f"with sample rate {sample_rate} Hz"
                )

            # Parse the numpy data
            # gwosc.datasets.fetch_open_data with format='numpy' returns
            # a tuple of (times, strains)
            times, strains = urls

        except Exception as e:
            raise RuntimeError(
                f"Failed to download strain data for detector '{detector}' "
                f"at GPS time {center_gps}. Error: {e}"
            ) from e

        # Convert to numpy arrays
        strain_array = np.array(strains, dtype=float)
        time_array = np.array(times, dtype=float)

        # Create time axis relative to center GPS time
        time_relative = time_array - center_gps

        # Build result dictionary
        result: dict[str, Any] = {
            "strain": DimArray(strain_array, unit=strain),
            "time": DimArray(time_relative, unit=second),
            "gps_start": gps_start,
            "gps_center": center_gps,
            "sample_rate": DimArray(np.array([sample_rate]), unit=hertz),
            "detector": detector,
        }

        if event is not None:
            result["event"] = event

        return result
