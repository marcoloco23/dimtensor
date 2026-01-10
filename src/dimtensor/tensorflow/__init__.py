"""TensorFlow integration for dimtensor.

Provides DimTensor and DimVariable, TensorFlow tensor/variable wrappers
with physical unit tracking. Supports both eager and graph (@tf.function)
execution modes.

Example:
    >>> import tensorflow as tf
    >>> from dimtensor.tensorflow import DimTensor, DimVariable
    >>> from dimtensor import units
    >>>
    >>> # Unit-aware tensors
    >>> velocity = DimTensor(tf.constant([1.0, 2.0, 3.0]), units.m / units.s)
    >>> time = DimTensor([0.5, 1.0, 1.5], units.s)
    >>> distance = velocity * time  # Result in meters
    >>>
    >>> # Trainable variables with units
    >>> position = DimVariable([0.0, 0.0, 0.0], units.m, name='position')
    >>> position.assign([1.0, 2.0, 3.0])
    >>>
    >>> # Works with @tf.function
    >>> @tf.function
    ... def compute_energy(mass, velocity):
    ...     return 0.5 * mass * velocity**2
    >>>
    >>> # Gradient computation
    >>> with tf.GradientTape() as tape:
    ...     energy = 0.5 * mass * (velocity ** 2)
    ...     loss = energy.sum()
    >>> grads = tape.gradient(loss.data, [mass.data])
"""

from .dimtensor import DimTensor
from .dimvariable import DimVariable

__all__ = ["DimTensor", "DimVariable"]
