"""JAX integration for dimtensor.

Provides JAX-compatible DimArray with pytree registration, JIT support,
vmap support, and grad support.

Example:
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from dimtensor.jax import DimArray
    >>> from dimtensor import units
    >>>
    >>> @jax.jit
    ... def compute_energy(mass, velocity):
    ...     return 0.5 * mass * velocity**2
    >>>
    >>> m = DimArray(jnp.array([1.0, 2.0]), units.kg)
    >>> v = DimArray(jnp.array([3.0, 4.0]), units.m / units.s)
    >>> energy = compute_energy(m, v)  # JIT preserves units
"""

from .dimarray import DimArray, register_pytree

# Note: register_pytree() is called automatically when dimarray.py is imported

__all__ = ["DimArray", "register_pytree"]
