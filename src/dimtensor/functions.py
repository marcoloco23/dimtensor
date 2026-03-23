"""Module-level array functions for DimArray.

Provides NumPy-style functions that operate on DimArrays while
maintaining dimensional correctness.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .core.dimarray import DimArray
from .core.units import Unit
from .errors import DimensionError


def _check_same_dimension(arrays: Sequence[DimArray], operation: str) -> Unit:
    """Verify all arrays have same dimension, return the first unit.

    Args:
        arrays: Sequence of DimArrays to check.
        operation: Name of the operation (for error messages).

    Returns:
        The unit of the first array.

    Raises:
        ValueError: If arrays sequence is empty.
        DimensionError: If arrays have incompatible dimensions.
    """
    if not arrays:
        raise ValueError(f"Need at least one array for {operation}")

    first_unit = arrays[0]._unit
    first_dim = first_unit.dimension

    for arr in arrays[1:]:
        if arr._unit.dimension != first_dim:
            raise DimensionError.incompatible(
                first_dim, arr._unit.dimension, operation
            )

    return first_unit


def concatenate(
    arrays: Sequence[DimArray],
    axis: int = 0,
) -> DimArray:
    """Join a sequence of DimArrays along an existing axis.

    All arrays must have the same unit dimension. Arrays with compatible
    units (e.g., km and m) will be converted to the first array's unit.

    If any arrays have uncertainty, the uncertainties are concatenated.
    Arrays without uncertainty are treated as having zero uncertainty.

    Args:
        arrays: Sequence of DimArrays to concatenate.
        axis: The axis along which to concatenate (default: 0).

    Returns:
        Concatenated DimArray with the same unit as the first input array.

    Raises:
        DimensionError: If arrays have incompatible dimensions.
        ValueError: If arrays sequence is empty.

    Examples:
        >>> a = DimArray([1.0, 2.0], units.m)
        >>> b = DimArray([3.0, 4.0], units.m)
        >>> concatenate([a, b])
        DimArray([1. 2. 3. 4.], unit='m')
    """
    unit = _check_same_dimension(arrays, "concatenate")

    # Convert all arrays to the same unit before concatenating
    converted = [arr.to(unit) for arr in arrays]
    raw_arrays = [arr._data for arr in converted]

    result = np.concatenate(raw_arrays, axis=axis)

    # Handle uncertainty
    any_have_uncertainty = any(arr._uncertainty is not None for arr in converted)
    new_uncertainty = None
    if any_have_uncertainty:
        # If mixing with/without uncertainty, treat missing as zero
        unc_arrays = []
        for arr in converted:
            if arr._uncertainty is not None:
                unc_arrays.append(arr._uncertainty)
            else:
                unc_arrays.append(np.zeros_like(arr._data))
        new_uncertainty = np.concatenate(unc_arrays, axis=axis)

    return DimArray._from_data_and_unit(result, unit, new_uncertainty)


def stack(
    arrays: Sequence[DimArray],
    axis: int = 0,
) -> DimArray:
    """Stack a sequence of DimArrays along a new axis.

    All arrays must have the same unit dimension. Arrays with compatible
    units (e.g., km and m) will be converted to the first array's unit.

    If any arrays have uncertainty, the uncertainties are stacked.
    Arrays without uncertainty are treated as having zero uncertainty.

    Args:
        arrays: Sequence of DimArrays to stack.
        axis: The axis along which to stack (default: 0).

    Returns:
        Stacked DimArray with the same unit as the first input array.

    Raises:
        DimensionError: If arrays have incompatible dimensions.
        ValueError: If arrays sequence is empty.

    Examples:
        >>> a = DimArray([1.0, 2.0], units.m)
        >>> b = DimArray([3.0, 4.0], units.m)
        >>> stack([a, b])
        DimArray([[1. 2.] [3. 4.]], unit='m')
    """
    unit = _check_same_dimension(arrays, "stack")

    # Convert all arrays to the same unit before stacking
    converted = [arr.to(unit) for arr in arrays]
    raw_arrays = [arr._data for arr in converted]

    result = np.stack(raw_arrays, axis=axis)

    # Handle uncertainty
    any_have_uncertainty = any(arr._uncertainty is not None for arr in converted)
    new_uncertainty = None
    if any_have_uncertainty:
        unc_arrays = []
        for arr in converted:
            if arr._uncertainty is not None:
                unc_arrays.append(arr._uncertainty)
            else:
                unc_arrays.append(np.zeros_like(arr._data))
        new_uncertainty = np.stack(unc_arrays, axis=axis)

    return DimArray._from_data_and_unit(result, unit, new_uncertainty)


def split(
    array: DimArray,
    indices_or_sections: int | Sequence[int],
    axis: int = 0,
) -> list[DimArray]:
    """Split a DimArray into sub-arrays.

    Args:
        array: The DimArray to split.
        indices_or_sections: If an integer N, split into N equal parts.
            If a sequence of integers, split at those indices.
        axis: The axis along which to split (default: 0).

    Returns:
        List of DimArrays, all with the same unit and uncertainty as the input.

    Examples:
        >>> arr = DimArray([1.0, 2.0, 3.0, 4.0], units.m)
        >>> parts = split(arr, 2)
        >>> len(parts)
        2
    """
    raw_splits = np.split(array._data, indices_or_sections, axis=axis)

    # Split uncertainty if present
    if array._uncertainty is not None:
        unc_splits = np.split(array._uncertainty, indices_or_sections, axis=axis)
        return [
            DimArray._from_data_and_unit(sub, array._unit, unc)
            for sub, unc in zip(raw_splits, unc_splits)
        ]
    else:
        return [
            DimArray._from_data_and_unit(sub, array._unit, None)
            for sub in raw_splits
        ]


def _propagate_bilinear_uncertainty(
    a: DimArray, b: DimArray
) -> NDArray[Any] | None:
    """Propagate uncertainty through a bilinear operation (dot/matmul).

    Assumes independent inputs. For C = A @ B:
        σ²(C_ij) = Σ_k (B_kj² σ²(A_ik) + A_ik² σ²(B_kj))

    Returns None if neither input has uncertainty.
    """
    if a._uncertainty is None and b._uncertainty is None:
        return None

    a_var = a._uncertainty**2 if a._uncertainty is not None else np.zeros_like(a._data)
    b_var = b._uncertainty**2 if b._uncertainty is not None else np.zeros_like(b._data)

    # variance(result) = (b²) @ (a_var) + (a²) @ (b_var)  — contracted over shared axis
    # np.dot/matmul contract the last axis of a with the second-to-last of b,
    # so the same contraction on squared values gives the variance.
    if a._data.ndim == 1 and b._data.ndim == 1:
        # 1D dot 1D -> scalar
        result_var = np.sum(b._data**2 * a_var + a._data**2 * b_var)
        return np.array([np.sqrt(result_var)])

    # General case: use matmul rules
    result_var = np.matmul(a_var, b._data**2) + np.matmul(a._data**2, b_var)
    return np.sqrt(result_var)  # type: ignore[no-any-return]


def dot(a: DimArray, b: DimArray) -> DimArray:
    """Dot product of two DimArrays.

    Dimensions multiply: if a has dimension D1 and b has dimension D2,
    the result has dimension D1 * D2.

    Uncertainty is propagated assuming independent inputs:
        σ² = Σ_k (b_k² σ²(a_k) + a_k² σ²(b_k))

    Args:
        a: First array.
        b: Second array.

    Returns:
        Dot product with multiplied dimensions.

    Examples:
        >>> length = DimArray([1.0, 2.0, 3.0], units.m)
        >>> force = DimArray([4.0, 5.0, 6.0], units.N)
        >>> work = dot(length, force)  # Result has dimension of energy (J)
    """
    result = np.dot(a._data, b._data)
    new_unit = a._unit * b._unit

    # Ensure result is at least 1D for API consistency
    if np.isscalar(result):
        result = np.array([result])

    # Uncertainty propagation (assumes independent inputs):
    # For 1D·1D: σ² = Σ_k (b_k² σ_a_k² + a_k² σ_b_k²)
    new_uncertainty = _propagate_bilinear_uncertainty(a, b)

    return DimArray._from_data_and_unit(result, new_unit, new_uncertainty)


def matmul(a: DimArray, b: DimArray) -> DimArray:
    """Matrix multiplication of two DimArrays.

    Dimensions multiply: if a has dimension D1 and b has dimension D2,
    the result has dimension D1 * D2.

    Uncertainty is propagated assuming independent inputs:
        σ²(C_ij) = Σ_k (B_kj² σ²(A_ik) + A_ik² σ²(B_kj))

    Args:
        a: First array (must be at least 1D).
        b: Second array (must be at least 1D).

    Returns:
        Matrix product with multiplied dimensions.

    Examples:
        >>> A = DimArray([[1, 2], [3, 4]], units.m)
        >>> B = DimArray([[5, 6], [7, 8]], units.s)
        >>> C = matmul(A, B)  # Result has dimension m*s
    """
    result = np.matmul(a._data, b._data)
    new_unit = a._unit * b._unit

    # Ensure result is at least 1D for API consistency
    if np.isscalar(result):
        result = np.array([result])

    # Uncertainty propagation (assumes independent inputs):
    # σ²(C_ij) = Σ_k (B_kj² σ_A_ik² + A_ik² σ_B_kj²)
    new_uncertainty = _propagate_bilinear_uncertainty(a, b)

    return DimArray._from_data_and_unit(result, new_unit, new_uncertainty)


def norm(
    array: DimArray,
    ord: float | None = None,
    axis: int | None = None,
    keepdims: bool = False,
) -> DimArray:
    """Compute the norm of a DimArray.

    The result preserves the original unit (norm of meters is meters).

    Note: Uncertainty propagation through norm is complex and not
    implemented. The result will have no uncertainty information.

    Args:
        array: Input array.
        ord: Order of the norm (see numpy.linalg.norm).
            None = 2-norm for vectors, Frobenius for matrices.
            Other values: 1, 2, inf, -inf, etc.
        axis: Axis along which to compute the norm.
            If None, computes norm of flattened array.
        keepdims: If True, the reduced axes are kept with size 1.

    Returns:
        Norm with the same unit as the input array.

    Examples:
        >>> v = DimArray([3.0, 4.0], units.m)
        >>> norm(v)  # 5.0 m
        DimArray([5.], unit='m')
    """
    result = np.linalg.norm(array._data, ord=ord, axis=axis, keepdims=keepdims)

    # Ensure result is at least 1D for API consistency
    result_arr = np.atleast_1d(result)

    # Uncertainty propagation through norm is complex, drop it
    return DimArray._from_data_and_unit(result_arr, array._unit, None)


def weighted_mean(
    arrays: Sequence[DimArray],
) -> DimArray:
    """Inverse-variance weighted mean of DimArrays.

    Computes the optimal combination of measurements with different
    uncertainties. Each value is weighted by 1/σ², giving more weight
    to more precise measurements.

    If any input has zero uncertainty, that value is returned directly
    (it has infinite weight, i.e., it is exact).

    All arrays must be scalar (single-element) and have the same dimension.
    All arrays must have uncertainty.

    Args:
        arrays: Sequence of scalar DimArrays with uncertainty.

    Returns:
        Weighted mean with propagated uncertainty σ = 1/√(Σ 1/σ_i²).

    Raises:
        DimensionError: If arrays have incompatible dimensions.
        ValueError: If arrays is empty or any array lacks uncertainty.

    Examples:
        >>> a = DimArray([10.0], units.m, uncertainty=[1.0])
        >>> b = DimArray([12.0], units.m, uncertainty=[2.0])
        >>> weighted_mean([a, b])  # Weighted toward a (smaller uncertainty)
        DimArray([10.4], unit='m')  # with uncertainty ~0.894
    """
    unit = _check_same_dimension(arrays, "weighted_mean")

    # Convert all to same unit
    converted = [arr.to(unit) for arr in arrays]

    for arr in converted:
        if arr._uncertainty is None:
            raise ValueError(
                "All arrays must have uncertainty for weighted_mean. "
                "Use mean() for unweighted averaging."
            )

    values = np.array([arr._data.item() for arr in converted])
    sigmas = np.array([arr._uncertainty.item() for arr in converted])  # type: ignore[union-attr]

    # Handle zero-variance (exact) inputs: return that value
    exact_mask = sigmas == 0.0
    if np.any(exact_mask):
        exact_values = values[exact_mask]
        return DimArray._from_data_and_unit(
            np.array([exact_values[0]]), unit, np.array([0.0])
        )

    weights = 1.0 / sigmas**2
    total_weight = np.sum(weights)
    result = np.sum(weights * values) / total_weight
    result_sigma = 1.0 / np.sqrt(total_weight)

    return DimArray._from_data_and_unit(
        np.array([result]), unit, np.array([result_sigma])
    )
