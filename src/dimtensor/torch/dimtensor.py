"""DimTensor: PyTorch tensors with dimensional awareness.

DimTensor wraps torch.Tensor and tracks physical units through all operations,
catching dimensional errors while preserving autograd functionality.
"""

from __future__ import annotations

import builtins
from typing import Any, Sequence

import torch
from torch import Tensor

from ..core.dimensions import Dimension
from ..core.units import Unit, dimensionless
from ..errors import DimensionError, UnitConversionError


class DimTensor:
    """A PyTorch tensor with attached physical units.

    DimTensor wraps a torch.Tensor and tracks its physical dimensions through
    all arithmetic operations. Operations between incompatible dimensions
    raise DimensionError immediately.

    Supports autograd - gradients flow through unit-aware operations.

    Examples:
        >>> import torch
        >>> from dimtensor.torch import DimTensor
        >>> from dimtensor import units
        >>>
        >>> v = DimTensor(torch.tensor([1.0, 2.0, 3.0]), units.m / units.s)
        >>> t = DimTensor(torch.tensor([0.5, 1.0, 1.5]), units.s)
        >>> d = v * t  # distance in meters
        >>> print(d)
        DimTensor([0.5, 2.0, 4.5], unit='m')
    """

    __slots__ = ("_data", "_unit")

    def __init__(
        self,
        data: Tensor | Sequence[float] | float,
        unit: Unit | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        requires_grad: bool = False,
    ) -> None:
        """Create a DimTensor.

        Args:
            data: Tensor data (torch.Tensor, list, or scalar).
            unit: Physical unit. If None, assumes dimensionless.
            dtype: Torch dtype (float32, float64, etc.).
            device: Device to place tensor on ('cpu', 'cuda', 'mps').
            requires_grad: Whether to track gradients.
        """
        if isinstance(data, DimTensor):
            tensor = data._data.clone()
            unit = unit if unit is not None else data._unit
        elif isinstance(data, Tensor):
            tensor = data.clone() if data.requires_grad else data.detach().clone()
        else:
            tensor = torch.tensor(data)

        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        if device is not None:
            tensor = tensor.to(device=device)
        if requires_grad:
            tensor = tensor.requires_grad_(True)

        self._data: Tensor = tensor
        self._unit: Unit = unit if unit is not None else dimensionless

    @classmethod
    def _from_tensor_and_unit(cls, tensor: Tensor, unit: Unit) -> DimTensor:
        """Internal constructor that doesn't copy tensor."""
        result = object.__new__(cls)
        result._data = tensor
        result._unit = unit
        return result

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def data(self) -> Tensor:
        """The underlying torch.Tensor."""
        return self._data

    @property
    def unit(self) -> Unit:
        """The physical unit of this tensor."""
        return self._unit

    @property
    def dimension(self) -> Dimension:
        """The physical dimension of this tensor."""
        return self._unit.dimension

    @property
    def shape(self) -> torch.Size:
        """Shape of the underlying tensor."""
        return self._data.shape

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self._data.ndim

    @property
    def numel(self) -> int:
        """Total number of elements."""
        return self._data.numel()

    @property
    def dtype(self) -> torch.dtype:
        """Data type of the underlying tensor."""
        return self._data.dtype

    @property
    def device(self) -> torch.device:
        """Device the tensor is on."""
        return self._data.device

    @property
    def requires_grad(self) -> bool:
        """Whether gradients are tracked."""
        return self._data.requires_grad

    @property
    def grad(self) -> Tensor | None:
        """Gradient of the tensor."""
        return self._data.grad

    @property
    def is_dimensionless(self) -> bool:
        """Check if this tensor is dimensionless."""
        return self._unit.dimension.is_dimensionless

    # =========================================================================
    # Device and dtype operations
    # =========================================================================

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> DimTensor:
        """Move tensor to device and/or cast dtype.

        For unit conversion, use to_unit() instead.

        Args:
            device: Target device ('cpu', 'cuda', 'mps').
            dtype: Target dtype (torch.float32, torch.float64, etc.).

        Returns:
            New DimTensor on target device/dtype.
        """
        new_tensor = self._data.to(device=device, dtype=dtype)
        return DimTensor._from_tensor_and_unit(new_tensor, self._unit)

    def cuda(self, device: int | None = None) -> DimTensor:
        """Move to CUDA device."""
        new_tensor = self._data.cuda(device)
        return DimTensor._from_tensor_and_unit(new_tensor, self._unit)

    def cpu(self) -> DimTensor:
        """Move to CPU."""
        new_tensor = self._data.cpu()
        return DimTensor._from_tensor_and_unit(new_tensor, self._unit)

    def float(self) -> DimTensor:
        """Cast to float32."""
        return DimTensor._from_tensor_and_unit(self._data.float(), self._unit)

    def double(self) -> DimTensor:
        """Cast to float64."""
        return DimTensor._from_tensor_and_unit(self._data.double(), self._unit)

    def half(self) -> DimTensor:
        """Cast to float16."""
        return DimTensor._from_tensor_and_unit(self._data.half(), self._unit)

    def bfloat16(self) -> DimTensor:
        """Cast to bfloat16."""
        return DimTensor._from_tensor_and_unit(self._data.bfloat16(), self._unit)

    # =========================================================================
    # Autograd operations
    # =========================================================================

    def requires_grad_(self, requires_grad: bool = True) -> DimTensor:
        """Set requires_grad in-place."""
        self._data.requires_grad_(requires_grad)
        return self

    def detach(self) -> DimTensor:
        """Return detached tensor."""
        return DimTensor._from_tensor_and_unit(self._data.detach(), self._unit)

    def backward(
        self,
        gradient: Tensor | None = None,
        retain_graph: bool | None = None,
        create_graph: bool = False,
    ) -> None:
        """Compute gradients."""
        self._data.backward(gradient, retain_graph, create_graph)  # type: ignore[no-untyped-call]

    # =========================================================================
    # Unit conversion
    # =========================================================================

    def to_unit(self, unit: Unit) -> DimTensor:
        """Convert to a different unit with the same dimension.

        Args:
            unit: Target unit (must have same dimension).

        Returns:
            New DimTensor with converted values.

        Raises:
            UnitConversionError: If dimensions don't match.
        """
        if not self._unit.is_compatible(unit):
            raise UnitConversionError.incompatible(self._unit.symbol, unit.symbol)

        factor = self._unit.conversion_factor(unit)
        new_tensor = self._data * factor
        return DimTensor._from_tensor_and_unit(new_tensor, unit)

    def magnitude(self) -> Tensor:
        """Return the numerical magnitude (stripping units).

        Use with caution - this loses dimensional safety.
        """
        return self._data.clone()

    # =========================================================================
    # Arithmetic operations
    # =========================================================================

    def __add__(self, other: DimTensor | Tensor | builtins.float) -> DimTensor:
        """Add tensors (must have same dimension)."""
        if isinstance(other, DimTensor):
            if self.dimension != other.dimension:
                raise DimensionError.incompatible(
                    self.dimension, other.dimension, "add"
                )
            other_converted = other.to_unit(self._unit)
            new_tensor = self._data + other_converted._data
            return DimTensor._from_tensor_and_unit(new_tensor, self._unit)
        else:
            if not self.is_dimensionless:
                raise DimensionError(
                    f"Cannot add dimensionless number to quantity with dimension {self.dimension}"
                )
            other_tensor = other if isinstance(other, Tensor) else torch.tensor(other)
            new_tensor = self._data + other_tensor
            return DimTensor._from_tensor_and_unit(new_tensor, self._unit)

    def __radd__(self, other: Tensor | builtins.float) -> DimTensor:
        """Right add."""
        return self.__add__(other)

    def __sub__(self, other: DimTensor | Tensor | builtins.float) -> DimTensor:
        """Subtract tensors (must have same dimension)."""
        if isinstance(other, DimTensor):
            if self.dimension != other.dimension:
                raise DimensionError.incompatible(
                    self.dimension, other.dimension, "subtract"
                )
            other_converted = other.to_unit(self._unit)
            new_tensor = self._data - other_converted._data
            return DimTensor._from_tensor_and_unit(new_tensor, self._unit)
        else:
            if not self.is_dimensionless:
                raise DimensionError(
                    f"Cannot subtract dimensionless number from quantity with dimension {self.dimension}"
                )
            other_tensor = other if isinstance(other, Tensor) else torch.tensor(other)
            new_tensor = self._data - other_tensor
            return DimTensor._from_tensor_and_unit(new_tensor, self._unit)

    def __rsub__(self, other: Tensor | builtins.float) -> DimTensor:
        """Right subtract."""
        if not self.is_dimensionless:
            raise DimensionError(
                f"Cannot subtract quantity with dimension {self.dimension} from dimensionless number"
            )
        other_tensor = other if isinstance(other, Tensor) else torch.tensor(other)
        new_tensor = other_tensor - self._data
        return DimTensor._from_tensor_and_unit(new_tensor, self._unit)

    def __mul__(self, other: DimTensor | Tensor | builtins.float) -> DimTensor:
        """Multiply tensors (dimensions multiply)."""
        if isinstance(other, DimTensor):
            new_unit = self._unit * other._unit
            new_tensor = self._data * other._data
            return DimTensor._from_tensor_and_unit(new_tensor, new_unit)
        else:
            other_tensor = other if isinstance(other, Tensor) else torch.tensor(other)
            new_tensor = self._data * other_tensor
            return DimTensor._from_tensor_and_unit(new_tensor, self._unit)

    def __rmul__(self, other: Tensor | builtins.float) -> DimTensor:
        """Right multiply."""
        return self.__mul__(other)

    def __truediv__(self, other: DimTensor | Tensor | builtins.float) -> DimTensor:
        """Divide tensors (dimensions divide)."""
        if isinstance(other, DimTensor):
            new_unit = self._unit / other._unit
            new_tensor = self._data / other._data
            return DimTensor._from_tensor_and_unit(new_tensor, new_unit)
        else:
            other_tensor = other if isinstance(other, Tensor) else torch.tensor(other)
            new_tensor = self._data / other_tensor
            return DimTensor._from_tensor_and_unit(new_tensor, self._unit)

    def __rtruediv__(self, other: Tensor | builtins.float) -> DimTensor:
        """Right divide."""
        new_unit = Unit(
            f"1/{self._unit.symbol}",
            self._unit.dimension ** -1,
            1.0 / self._unit.scale,
        )
        other_tensor = other if isinstance(other, Tensor) else torch.tensor(other)
        new_tensor = other_tensor / self._data
        return DimTensor._from_tensor_and_unit(new_tensor, new_unit)

    def __pow__(self, power: int | builtins.float) -> DimTensor:
        """Raise to a power."""
        new_unit = self._unit ** power
        new_tensor = self._data ** power
        return DimTensor._from_tensor_and_unit(new_tensor, new_unit)

    def __neg__(self) -> DimTensor:
        """Negate values."""
        return DimTensor._from_tensor_and_unit(-self._data, self._unit)

    def __pos__(self) -> DimTensor:
        """Unary positive."""
        return DimTensor._from_tensor_and_unit(+self._data, self._unit)

    def __abs__(self) -> DimTensor:
        """Absolute value."""
        return DimTensor._from_tensor_and_unit(torch.abs(self._data), self._unit)

    def sqrt(self) -> DimTensor:
        """Square root (dimension exponents halve)."""
        return self ** 0.5

    # =========================================================================
    # Comparison operations
    # =========================================================================

    def __eq__(self, other: object) -> Tensor | bool:  # type: ignore[override]
        """Element-wise equality."""
        if isinstance(other, DimTensor):
            if self.dimension != other.dimension:
                return False
            other_converted = other.to_unit(self._unit)
            return self._data == other_converted._data
        elif self.is_dimensionless:
            other_tensor = other if isinstance(other, Tensor) else torch.tensor(other)
            return self._data == other_tensor
        return False

    def __lt__(self, other: DimTensor | Tensor | builtins.float) -> Tensor:
        """Element-wise less than."""
        if isinstance(other, DimTensor):
            if self.dimension != other.dimension:
                raise DimensionError.incompatible(
                    self.dimension, other.dimension, "compare"
                )
            other_converted = other.to_unit(self._unit)
            return self._data < other_converted._data
        elif self.is_dimensionless:
            other_tensor = other if isinstance(other, Tensor) else torch.tensor(other)
            return self._data < other_tensor
        raise DimensionError(
            f"Cannot compare quantity with dimension {self.dimension} to dimensionless number"
        )

    def __le__(self, other: DimTensor | Tensor | builtins.float) -> Tensor:
        """Element-wise less than or equal."""
        if isinstance(other, DimTensor):
            if self.dimension != other.dimension:
                raise DimensionError.incompatible(
                    self.dimension, other.dimension, "compare"
                )
            other_converted = other.to_unit(self._unit)
            return self._data <= other_converted._data
        elif self.is_dimensionless:
            other_tensor = other if isinstance(other, Tensor) else torch.tensor(other)
            return self._data <= other_tensor
        raise DimensionError(
            f"Cannot compare quantity with dimension {self.dimension} to dimensionless number"
        )

    def __gt__(self, other: DimTensor | Tensor | builtins.float) -> Tensor:
        """Element-wise greater than."""
        if isinstance(other, DimTensor):
            if self.dimension != other.dimension:
                raise DimensionError.incompatible(
                    self.dimension, other.dimension, "compare"
                )
            other_converted = other.to_unit(self._unit)
            return self._data > other_converted._data
        elif self.is_dimensionless:
            other_tensor = other if isinstance(other, Tensor) else torch.tensor(other)
            return self._data > other_tensor
        raise DimensionError(
            f"Cannot compare quantity with dimension {self.dimension} to dimensionless number"
        )

    def __ge__(self, other: DimTensor | Tensor | builtins.float) -> Tensor:
        """Element-wise greater than or equal."""
        if isinstance(other, DimTensor):
            if self.dimension != other.dimension:
                raise DimensionError.incompatible(
                    self.dimension, other.dimension, "compare"
                )
            other_converted = other.to_unit(self._unit)
            return self._data >= other_converted._data
        elif self.is_dimensionless:
            other_tensor = other if isinstance(other, Tensor) else torch.tensor(other)
            return self._data >= other_tensor
        raise DimensionError(
            f"Cannot compare quantity with dimension {self.dimension} to dimensionless number"
        )

    # =========================================================================
    # Indexing
    # =========================================================================

    def __getitem__(self, key: Any) -> DimTensor:
        """Index into the tensor, preserving units."""
        result = self._data[key]
        return DimTensor._from_tensor_and_unit(result, self._unit)

    def __len__(self) -> int:
        """Length of first dimension."""
        return len(self._data)

    # =========================================================================
    # Reduction operations
    # =========================================================================

    def sum(self, dim: int | None = None, keepdim: bool = False) -> DimTensor:
        """Sum of tensor elements."""
        result = self._data.sum(dim=dim, keepdim=keepdim)
        return DimTensor._from_tensor_and_unit(result, self._unit)

    def mean(self, dim: int | None = None, keepdim: bool = False) -> DimTensor:
        """Mean of tensor elements."""
        result = self._data.mean(dim=dim, keepdim=keepdim)
        return DimTensor._from_tensor_and_unit(result, self._unit)

    def std(self, dim: int | None = None, keepdim: bool = False) -> DimTensor:
        """Standard deviation of tensor elements."""
        result = self._data.std(dim=dim, keepdim=keepdim)
        return DimTensor._from_tensor_and_unit(result, self._unit)

    def var(self, dim: int | None = None, keepdim: bool = False) -> DimTensor:
        """Variance of tensor elements (squared units)."""
        result = self._data.var(dim=dim, keepdim=keepdim)
        new_unit = self._unit ** 2
        return DimTensor._from_tensor_and_unit(result, new_unit)

    def min(self, dim: int | None = None, keepdim: bool = False) -> DimTensor:
        """Minimum value."""
        if dim is None:
            result = self._data.min()
        else:
            result = self._data.min(dim=dim, keepdim=keepdim).values
        return DimTensor._from_tensor_and_unit(result, self._unit)

    def max(self, dim: int | None = None, keepdim: bool = False) -> DimTensor:
        """Maximum value."""
        if dim is None:
            result = self._data.max()
        else:
            result = self._data.max(dim=dim, keepdim=keepdim).values
        return DimTensor._from_tensor_and_unit(result, self._unit)

    def norm(
        self,
        p: builtins.float | str = 2,
        dim: int | tuple[int, ...] | None = None,
        keepdim: bool = False,
    ) -> DimTensor:
        """Vector or matrix norm (preserves units for p=1,2,inf)."""
        result = torch.linalg.norm(self._data, ord=p, dim=dim, keepdim=keepdim)
        return DimTensor._from_tensor_and_unit(result, self._unit)

    # =========================================================================
    # Reshaping operations
    # =========================================================================

    def reshape(self, *shape: int) -> DimTensor:
        """Reshape tensor preserving units."""
        new_tensor = self._data.reshape(*shape)
        return DimTensor._from_tensor_and_unit(new_tensor, self._unit)

    def view(self, *shape: int) -> DimTensor:
        """View tensor with new shape."""
        new_tensor = self._data.view(*shape)
        return DimTensor._from_tensor_and_unit(new_tensor, self._unit)

    def transpose(self, dim0: int, dim1: int) -> DimTensor:
        """Transpose two dimensions."""
        new_tensor = self._data.transpose(dim0, dim1)
        return DimTensor._from_tensor_and_unit(new_tensor, self._unit)

    def permute(self, *dims: int) -> DimTensor:
        """Permute dimensions."""
        new_tensor = self._data.permute(*dims)
        return DimTensor._from_tensor_and_unit(new_tensor, self._unit)

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> DimTensor:
        """Flatten dimensions."""
        new_tensor = self._data.flatten(start_dim, end_dim)
        return DimTensor._from_tensor_and_unit(new_tensor, self._unit)

    def squeeze(self, dim: int | None = None) -> DimTensor:
        """Remove size-1 dimensions."""
        if dim is None:
            new_tensor = self._data.squeeze()
        else:
            new_tensor = self._data.squeeze(dim)
        return DimTensor._from_tensor_and_unit(new_tensor, self._unit)

    def unsqueeze(self, dim: int) -> DimTensor:
        """Add size-1 dimension."""
        new_tensor = self._data.unsqueeze(dim)
        return DimTensor._from_tensor_and_unit(new_tensor, self._unit)

    # =========================================================================
    # Linear algebra
    # =========================================================================

    def matmul(self, other: DimTensor) -> DimTensor:
        """Matrix multiplication (dimensions multiply)."""
        new_unit = self._unit * other._unit
        new_tensor = torch.matmul(self._data, other._data)
        return DimTensor._from_tensor_and_unit(new_tensor, new_unit)

    def __matmul__(self, other: DimTensor) -> DimTensor:
        """Matrix multiplication operator @."""
        return self.matmul(other)

    def dot(self, other: DimTensor) -> DimTensor:
        """Dot product (dimensions multiply)."""
        if self._data.ndim != 1 or other._data.ndim != 1:
            raise ValueError("dot requires 1D tensors")
        new_unit = self._unit * other._unit
        new_tensor = torch.dot(self._data, other._data)
        return DimTensor._from_tensor_and_unit(new_tensor, new_unit)

    # =========================================================================
    # String representations
    # =========================================================================

    def __repr__(self) -> str:
        """Detailed representation."""
        return f"DimTensor({self._data!r}, unit={self._unit.symbol!r})"

    def __str__(self) -> str:
        """Human-readable string."""
        if self.is_dimensionless:
            return str(self._data)
        simplified = self._unit.simplified()
        return f"{self._data} {simplified.symbol}"

    # =========================================================================
    # Conversion
    # =========================================================================

    def numpy(self) -> Any:
        """Convert to numpy array (loses unit information)."""
        return self._data.detach().cpu().numpy()

    def item(self) -> builtins.float:
        """Get single-element tensor as Python scalar."""
        return self._data.item()
