"""Conservation law tracking for physics simulations.

Track conserved quantities (energy, momentum, mass) across computations
and verify conservation within a tolerance.

Usage:
    >>> from dimtensor import DimArray, units
    >>> from dimtensor.validation import ConservationTracker
    >>>
    >>> # Track energy conservation
    >>> tracker = ConservationTracker("Total Energy")
    >>> tracker.record(initial_energy)
    >>> # ... do computation ...
    >>> tracker.record(final_energy)
    >>> if not tracker.is_conserved(rtol=1e-6):
    ...     print(f"Energy drift: {tracker.drift()}")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..core.dimarray import DimArray


class ConservationTracker:
    """Track a conserved quantity across computations.

    Records snapshots of a value and checks if it remains constant
    within a specified tolerance.

    Attributes:
        name: Human-readable name of the conserved quantity.
        history: List of recorded values.

    Example:
        >>> from dimtensor import DimArray, units
        >>> from dimtensor.validation import ConservationTracker
        >>>
        >>> tracker = ConservationTracker("Total Energy")
        >>> E1 = DimArray([100.0], units.J)
        >>> tracker.record(E1)
        >>> E2 = DimArray([99.9999], units.J)
        >>> tracker.record(E2)
        >>> tracker.is_conserved(rtol=1e-4)  # True
        >>> tracker.is_conserved(rtol=1e-6)  # False
    """

    def __init__(self, name: str = "Conserved Quantity") -> None:
        """Create a conservation tracker.

        Args:
            name: Human-readable name of the quantity being tracked.
        """
        self.name = name
        self.history: list[float] = []
        self._unit_symbol: str | None = None

    def record(self, value: DimArray | float | np.ndarray[Any, Any]) -> None:
        """Record a checkpoint value.

        Args:
            value: The current value of the conserved quantity.
                If a DimArray, uses the scalar value and tracks the unit.
                If an array, sums all elements.

        Raises:
            ValueError: If the value has incompatible units with previous records.
        """
        # Import here to avoid circular dependency
        from ..core.dimarray import DimArray

        if isinstance(value, DimArray):
            # Extract scalar or sum
            if value.size == 1:
                scalar = float(value.data.item())
            else:
                scalar = float(value.data.sum())

            # Track unit for consistency checking
            unit_symbol = value.unit.symbol
            if self._unit_symbol is None:
                self._unit_symbol = unit_symbol
            elif self._unit_symbol != unit_symbol:
                raise ValueError(
                    f"Unit mismatch: expected {self._unit_symbol}, got {unit_symbol}. "
                    f"Convert to consistent units before recording."
                )
        elif isinstance(value, np.ndarray):
            scalar = float(value.sum())
        else:
            scalar = float(value)

        self.history.append(scalar)

    def is_conserved(self, rtol: float = 1e-9, atol: float = 0.0) -> bool:
        """Check if the quantity is conserved within tolerance.

        Compares all recorded values to the first recorded value.

        Args:
            rtol: Relative tolerance (default: 1e-9).
            atol: Absolute tolerance (default: 0.0).

        Returns:
            True if all values are within tolerance of the first value.

        Raises:
            ValueError: If no values have been recorded.
        """
        if len(self.history) == 0:
            raise ValueError("No values recorded. Call record() first.")
        if len(self.history) == 1:
            return True  # Single value is trivially conserved

        initial = self.history[0]
        for value in self.history[1:]:
            if not np.isclose(value, initial, rtol=rtol, atol=atol):
                return False
        return True

    def drift(self) -> float:
        """Calculate the relative drift from initial value.

        Returns:
            (current - initial) / |initial|, or absolute drift if initial is zero.

        Raises:
            ValueError: If fewer than 2 values have been recorded.
        """
        if len(self.history) < 2:
            raise ValueError("Need at least 2 recorded values to calculate drift.")

        initial = self.history[0]
        current = self.history[-1]

        if abs(initial) < 1e-15:
            # Avoid division by zero; return absolute drift
            return current - initial
        return (current - initial) / abs(initial)

    def max_drift(self) -> float:
        """Calculate the maximum relative drift from initial value.

        Returns:
            Maximum of |(value - initial) / |initial|| across all recorded values.

        Raises:
            ValueError: If fewer than 2 values have been recorded.
        """
        if len(self.history) < 2:
            raise ValueError("Need at least 2 recorded values to calculate drift.")

        initial = self.history[0]
        if abs(initial) < 1e-15:
            # Return max absolute drift
            return max(abs(v - initial) for v in self.history[1:])

        return max(abs((v - initial) / abs(initial)) for v in self.history[1:])

    def reset(self) -> None:
        """Clear the history and start fresh."""
        self.history.clear()
        self._unit_symbol = None

    def __repr__(self) -> str:
        """String representation."""
        unit_str = f" ({self._unit_symbol})" if self._unit_symbol else ""
        return (
            f"ConservationTracker('{self.name}'{unit_str}, "
            f"records={len(self.history)})"
        )

    def __len__(self) -> int:
        """Return number of recorded values."""
        return len(self.history)
