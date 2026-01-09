"""NIST CODATA fundamental constants loader.

Loads physical constants from NIST CODATA 2022 values.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ...core.dimarray import DimArray
from ...core.dimensions import Dimension
from ...core.units import (
    coulomb,
    electronvolt,
    joule,
    kelvin,
    kg,
    meter,
    mole,
    second,
)
from .base import BaseLoader


class NISTCODATALoader(BaseLoader):
    """Loader for NIST CODATA fundamental physical constants."""

    # NIST provides fundamental constants in ASCII format
    URL = "https://physics.nist.gov/cuu/Constants/Table/allascii.txt"

    def load(self, **kwargs: Any) -> dict[str, DimArray]:
        """Load NIST CODATA 2022 constants.

        Args:
            **kwargs: Additional arguments (force_download, etc.).

        Returns:
            Dictionary mapping constant names to DimArray values.

        Example:
            >>> loader = NISTCODATALoader()
            >>> constants = loader.load()
            >>> c = constants['speed of light in vacuum']
            >>> print(c)
        """
        force = kwargs.get("force_download", False)

        try:
            # Download the constants file
            filepath = self.download(
                self.URL,
                cache_key="nist_codata_2022",
                force=force,
            )

            # Parse the file
            constants = self._parse_codata_file(filepath)

            # If parsing failed or returned empty, use fallback
            if not constants:
                constants = self._get_fallback_constants()

        except (RuntimeError, OSError, IOError):
            # If download fails, use fallback constants
            constants = self._get_fallback_constants()

        return constants

    def _parse_codata_file(self, filepath: Path) -> dict[str, DimArray]:
        """Parse NIST CODATA ASCII file.

        The format is:
        Quantity                                Value            Uncertainty  Unit
        ----------------------------------------------------------------------------
        ...

        Args:
            filepath: Path to downloaded file.

        Returns:
            Dictionary of constant names to DimArray values.
        """
        content = filepath.read_text()
        lines = content.split("\n")

        constants = {}

        # Find the data section (after the header)
        data_start = None
        for i, line in enumerate(lines):
            if "----" in line:
                data_start = i + 1
                break

        if data_start is None:
            # Fallback: try parsing all lines
            data_start = 0

        # Parse each line
        for line in lines[data_start:]:
            if not line.strip():
                continue

            # Try to extract name, value, uncertainty, unit
            # Format: name (sometimes multi-word) + value + uncertainty + unit
            match = re.match(
                r"^([\w\s\-,()]+?)\s+([\d.eE+-]+)\s+([\d.eE+-]+|exact|\(exact\)|\.\.\.)\s*(.*)$",
                line.strip(),
            )

            if match:
                name = match.group(1).strip()
                value_str = match.group(2).strip()
                uncertainty_str = match.group(3).strip()
                unit_str = match.group(4).strip()

                try:
                    value = float(value_str)
                except ValueError:
                    continue

                # Parse unit and create DimArray
                unit = self._parse_unit(unit_str)

                if unit is not None:
                    constants[name] = DimArray([value], unit=unit)

        # If parsing failed, add some hardcoded constants as fallback
        if not constants:
            constants = self._get_fallback_constants()

        return constants

    def _parse_unit(self, unit_str: str) -> Any:
        """Parse unit string to dimtensor unit.

        Args:
            unit_str: Unit string (e.g., "m s^-1", "J K^-1").

        Returns:
            Unit object or None if cannot parse.
        """
        if not unit_str or unit_str == "":
            return None

        # Simple unit mapping
        unit_map = {
            "m": meter,
            "s": second,
            "kg": kg,
            "K": kelvin,
            "J": joule,
            "C": coulomb,
            "eV": electronvolt,
            "mol": mole,
        }

        # Try exact match first
        if unit_str in unit_map:
            return unit_map[unit_str]

        # Try parsing compound units (e.g., "m s^-1")
        # For now, return None for complex units
        # Future: implement full unit parser

        return None

    def _get_fallback_constants(self) -> dict[str, DimArray]:
        """Get hardcoded fallback constants if parsing fails.

        Returns:
            Dictionary of key fundamental constants as DimArrays.
        """
        import numpy as np
        from ...constants.universal import c, G, h, hbar

        # Convert Constant objects to DimArray
        return {
            "speed of light in vacuum": DimArray(np.array([c.value]), unit=c.unit),
            "Newtonian constant of gravitation": DimArray(np.array([G.value]), unit=G.unit),
            "Planck constant": DimArray(np.array([h.value]), unit=h.unit),
            "reduced Planck constant": DimArray(np.array([hbar.value]), unit=hbar.unit),
        }
