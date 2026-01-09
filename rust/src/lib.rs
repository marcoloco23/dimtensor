//! dimtensor_core: Rust backend for dimtensor
//!
//! This module provides accelerated operations for dimtensor, a Python library
//! for unit-aware tensors. It implements dimension checking and array operations
//! in Rust for improved performance.
//!
//! The module is designed to be used as an optional backend - if not available,
//! dimtensor falls back to pure Python implementations.

use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyTypeError};
use num_rational::Rational32;

mod dimension;
mod ops;

pub use dimension::RustDimension;

/// Check if two dimensions are compatible for addition/subtraction.
///
/// Dimensions are compatible if all 7 exponents are equal.
#[pyfunction]
fn dimensions_compatible(dim_a: &RustDimension, dim_b: &RustDimension) -> bool {
    dim_a.is_compatible(dim_b)
}

/// Multiply two dimensions (for multiplication operations).
///
/// When multiplying quantities, their dimensions add:
/// (m/s) * (s) = m
#[pyfunction]
fn multiply_dimensions(dim_a: &RustDimension, dim_b: &RustDimension) -> RustDimension {
    dim_a.multiply(dim_b)
}

/// Divide two dimensions (for division operations).
///
/// When dividing quantities, their dimensions subtract:
/// (m) / (s) = m/s
#[pyfunction]
fn divide_dimensions(dim_a: &RustDimension, dim_b: &RustDimension) -> RustDimension {
    dim_a.divide(dim_b)
}

/// Raise a dimension to an integer power.
///
/// When raising to a power, dimensions scale:
/// (m)^2 = m^2
#[pyfunction]
fn power_dimension(dim: &RustDimension, power: i32) -> RustDimension {
    dim.power(power)
}

/// Add two arrays with dimension checking.
///
/// Returns the result array. Raises ValueError if dimensions are incompatible.
#[pyfunction]
fn add_arrays<'py>(
    py: Python<'py>,
    a: PyReadonlyArrayDyn<'py, f64>,
    b: PyReadonlyArrayDyn<'py, f64>,
    dim_a: &RustDimension,
    dim_b: &RustDimension,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    if !dim_a.is_compatible(dim_b) {
        return Err(PyValueError::new_err(format!(
            "Cannot add arrays with incompatible dimensions: {} and {}",
            dim_a, dim_b
        )));
    }

    let arr_a = a.as_array();
    let arr_b = b.as_array();

    // Perform addition
    let result = &arr_a + &arr_b;

    Ok(result.into_pyarray(py))
}

/// Subtract two arrays with dimension checking.
///
/// Returns the result array. Raises ValueError if dimensions are incompatible.
#[pyfunction]
fn sub_arrays<'py>(
    py: Python<'py>,
    a: PyReadonlyArrayDyn<'py, f64>,
    b: PyReadonlyArrayDyn<'py, f64>,
    dim_a: &RustDimension,
    dim_b: &RustDimension,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    if !dim_a.is_compatible(dim_b) {
        return Err(PyValueError::new_err(format!(
            "Cannot subtract arrays with incompatible dimensions: {} and {}",
            dim_a, dim_b
        )));
    }

    let arr_a = a.as_array();
    let arr_b = b.as_array();

    let result = &arr_a - &arr_b;

    Ok(result.into_pyarray(py))
}

/// Multiply two arrays, combining their dimensions.
///
/// Returns (result_array, result_dimension).
#[pyfunction]
fn mul_arrays<'py>(
    py: Python<'py>,
    a: PyReadonlyArrayDyn<'py, f64>,
    b: PyReadonlyArrayDyn<'py, f64>,
    dim_a: &RustDimension,
    dim_b: &RustDimension,
) -> PyResult<(Bound<'py, PyArrayDyn<f64>>, RustDimension)> {
    let arr_a = a.as_array();
    let arr_b = b.as_array();

    let result = &arr_a * &arr_b;
    let result_dim = dim_a.multiply(dim_b);

    Ok((result.into_pyarray(py), result_dim))
}

/// Divide two arrays, combining their dimensions.
///
/// Returns (result_array, result_dimension).
#[pyfunction]
fn div_arrays<'py>(
    py: Python<'py>,
    a: PyReadonlyArrayDyn<'py, f64>,
    b: PyReadonlyArrayDyn<'py, f64>,
    dim_a: &RustDimension,
    dim_b: &RustDimension,
) -> PyResult<(Bound<'py, PyArrayDyn<f64>>, RustDimension)> {
    let arr_a = a.as_array();
    let arr_b = b.as_array();

    let result = &arr_a / &arr_b;
    let result_dim = dim_a.divide(dim_b);

    Ok((result.into_pyarray(py), result_dim))
}

/// The main Python module for dimtensor's Rust backend.
#[pymodule]
fn dimtensor_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustDimension>()?;
    m.add_function(wrap_pyfunction!(dimensions_compatible, m)?)?;
    m.add_function(wrap_pyfunction!(multiply_dimensions, m)?)?;
    m.add_function(wrap_pyfunction!(divide_dimensions, m)?)?;
    m.add_function(wrap_pyfunction!(power_dimension, m)?)?;
    m.add_function(wrap_pyfunction!(add_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(sub_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(mul_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(div_arrays, m)?)?;

    // Add version info
    m.add("__version__", "0.1.0")?;
    m.add("HAS_RUST_BACKEND", true)?;

    Ok(())
}
