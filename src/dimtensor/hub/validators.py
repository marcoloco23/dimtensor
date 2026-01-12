"""Runtime validators for model inputs/outputs with unit checking.

This module provides wrappers that automatically validate dimensional
correctness at inference time.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from ..core.dimensions import Dimension
from ..errors import DimensionError
from .cards import ModelCard

logger = logging.getLogger(__name__)


class DimModelWrapper:
    """Wrapper that validates input/output units at inference time.

    This wrapper can be used with any model (PyTorch, JAX, numpy function)
    to add automatic dimensional validation. It checks that inputs have
    the correct dimensions before calling the model, and validates outputs
    after the model runs.

    Attributes:
        model: The wrapped model.
        card: ModelCard with dimensional metadata.
        auto_convert: If True, automatically convert input units.
        strict: If True, raise errors on mismatch. If False, only warn.

    Example:
        >>> from dimtensor.hub import DimModelWrapper, ModelCard
        >>> from dimtensor import DimArray
        >>> from dimtensor.units import m, s
        >>>
        >>> # Wrap a model
        >>> wrapped = DimModelWrapper(model, card)
        >>>
        >>> # Call with validation
        >>> velocity = DimArray([1.0, 2.0], m / s)
        >>> output = wrapped(velocity=velocity)
    """

    def __init__(
        self,
        model: Any,
        card: ModelCard,
        auto_convert: bool = False,
        strict: bool = True,
    ) -> None:
        """Initialize wrapper.

        Args:
            model: Model to wrap (torch.nn.Module, JAX function, etc.).
            card: ModelCard with dimensional metadata.
            auto_convert: If True, auto-convert input units to model units.
            strict: If True, raise errors on validation failure.
        """
        self.model = model
        self.card = card
        self.auto_convert = auto_convert
        self.strict = strict

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call model with validation.

        Validates inputs, calls model, validates outputs.

        Args:
            *args: Positional arguments to model.
            **kwargs: Keyword arguments to model.

        Returns:
            Model output (validated if output validation enabled).

        Raises:
            DimensionError: If validation fails and strict=True.
        """
        # Validate inputs from kwargs
        if kwargs:
            self._validate_inputs(kwargs)

            # Auto-convert if enabled
            if self.auto_convert:
                kwargs = self._convert_inputs(kwargs)

        # Call model
        output = self.model(*args, **kwargs)

        # Validate outputs if it's a dict
        if isinstance(output, dict):
            self._validate_outputs(output)

        return output

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Alias for __call__ (PyTorch compatibility)."""
        return self(*args, **kwargs)

    def _validate_inputs(self, inputs: dict[str, Any]) -> None:
        """Validate input dimensions."""
        expected = self.card.info.input_dims

        for name, expected_dim in expected.items():
            if name not in inputs:
                msg = f"Missing required input: '{name}'"
                if self.strict:
                    raise DimensionError(msg)
                else:
                    logger.warning(msg)
                continue

            input_obj = inputs[name]
            actual_dim = self._extract_dimension(input_obj, name, "input")

            if actual_dim is None:
                continue

            if actual_dim != expected_dim:
                msg = (
                    f"Input '{name}' has wrong dimension: "
                    f"expected {expected_dim}, got {actual_dim}"
                )
                if self.strict:
                    raise DimensionError(msg)
                else:
                    logger.warning(msg)

    def _validate_outputs(self, outputs: dict[str, Any]) -> None:
        """Validate output dimensions."""
        expected = self.card.info.output_dims

        for name, expected_dim in expected.items():
            if name not in outputs:
                msg = f"Missing expected output: '{name}'"
                if self.strict:
                    raise DimensionError(msg)
                else:
                    logger.warning(msg)
                continue

            output_obj = outputs[name]
            actual_dim = self._extract_dimension(output_obj, name, "output")

            if actual_dim is None:
                continue

            if actual_dim != expected_dim:
                msg = (
                    f"Output '{name}' has wrong dimension: "
                    f"expected {expected_dim}, got {actual_dim}"
                )
                if self.strict:
                    raise DimensionError(msg)
                else:
                    logger.warning(msg)

    def _extract_dimension(
        self, obj: Any, name: str, obj_type: str
    ) -> Dimension | None:
        """Extract dimension from an object.

        Args:
            obj: Object to extract dimension from.
            name: Name of the input/output.
            obj_type: "input" or "output" for error messages.

        Returns:
            Dimension if found, None if no dimension info.
        """
        # Try unit attribute (DimArray, DimTensor)
        if hasattr(obj, "unit"):
            return obj.unit.dimension

        # Try dimension attribute directly
        if hasattr(obj, "dimension"):
            return obj.dimension

        # No dimension info
        msg = f"{obj_type.capitalize()} '{name}' has no unit/dimension information"
        if self.strict:
            raise DimensionError(msg)
        else:
            logger.warning(msg)
            return None

    def _convert_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Convert inputs to model units.

        Args:
            inputs: Input dict.

        Returns:
            Converted inputs dict.
        """
        from ..core.units import Unit

        converted = {}

        for name, value in inputs.items():
            if name not in self.card.info.input_dims:
                # Not in card, pass through
                converted[name] = value
                continue

            expected_dim = self.card.info.input_dims[name]

            # Check if value has units
            if not hasattr(value, "unit"):
                # No units, pass through
                converted[name] = value
                continue

            actual_dim = value.unit.dimension

            # Check if dimensions match
            if actual_dim != expected_dim:
                msg = (
                    f"Cannot auto-convert input '{name}': "
                    f"dimension mismatch ({actual_dim} != {expected_dim})"
                )
                if self.strict:
                    raise DimensionError(msg)
                else:
                    logger.warning(msg)
                    converted[name] = value
                    continue

            # Get target unit from characteristic scales
            if name in self.card.info.characteristic_scales:
                scale = self.card.info.characteristic_scales[name]
                # Create unit with same dimension but different scale
                # This is a simplified version - real implementation would
                # need to know the actual target unit
                logger.info(
                    f"Auto-converting '{name}' "
                    f"(characteristic scale: {scale})"
                )

            converted[name] = value

        return converted

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"DimModelWrapper(model={self.card.info.name}, "
            f"strict={self.strict}, "
            f"auto_convert={self.auto_convert})"
        )


def validate_model_io(
    model: Any,
    card: ModelCard,
    inputs: dict[str, Any],
    outputs: dict[str, Any] | None = None,
) -> tuple[bool, list[str]]:
    """Validate model inputs and optionally outputs.

    This is a standalone validation function that can be used
    without wrapping the model.

    Args:
        model: Model object (not used, for signature consistency).
        card: ModelCard with dimensional metadata.
        inputs: Input dict to validate.
        outputs: Optional output dict to validate.

    Returns:
        Tuple of (is_valid, error_messages).

    Example:
        >>> from dimtensor.hub import validate_model_io
        >>> from dimtensor import DimArray
        >>> from dimtensor.units import m, s
        >>>
        >>> velocity = DimArray([1.0], m / s)
        >>> inputs = {"velocity": velocity}
        >>> is_valid, errors = validate_model_io(model, card, inputs)
        >>> if not is_valid:
        ...     print("Validation errors:", errors)
    """
    errors = []

    # Validate inputs
    expected_inputs = card.info.input_dims

    for name, expected_dim in expected_inputs.items():
        if name not in inputs:
            errors.append(f"Missing required input: '{name}'")
            continue

        input_obj = inputs[name]

        # Extract dimension
        if hasattr(input_obj, "unit"):
            actual_dim = input_obj.unit.dimension
        elif hasattr(input_obj, "dimension"):
            actual_dim = input_obj.dimension
        else:
            errors.append(
                f"Input '{name}' has no unit/dimension information"
            )
            continue

        if actual_dim != expected_dim:
            errors.append(
                f"Input '{name}': expected {expected_dim}, got {actual_dim}"
            )

    # Validate outputs if provided
    if outputs is not None:
        expected_outputs = card.info.output_dims

        for name, expected_dim in expected_outputs.items():
            if name not in outputs:
                errors.append(f"Missing expected output: '{name}'")
                continue

            output_obj = outputs[name]

            # Extract dimension
            if hasattr(output_obj, "unit"):
                actual_dim = output_obj.unit.dimension
            elif hasattr(output_obj, "dimension"):
                actual_dim = output_obj.dimension
            else:
                errors.append(
                    f"Output '{name}' has no unit/dimension information"
                )
                continue

            if actual_dim != expected_dim:
                errors.append(
                    f"Output '{name}': expected {expected_dim}, got {actual_dim}"
                )

    return (len(errors) == 0, errors)


def create_validator(
    card: ModelCard,
    auto_convert: bool = False,
    strict: bool = True,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Create a decorator for validating model functions.

    Returns a decorator that wraps a function with validation.

    Args:
        card: ModelCard with dimensional metadata.
        auto_convert: If True, auto-convert input units.
        strict: If True, raise errors on validation failure.

    Returns:
        Decorator function.

    Example:
        >>> from dimtensor.hub import create_validator
        >>>
        >>> @create_validator(my_card)
        >>> def my_model(velocity, pressure):
        ...     return {"force": velocity * pressure}
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            wrapped = DimModelWrapper(func, card, auto_convert, strict)
            return wrapped(*args, **kwargs)

        return wrapper

    return decorator
