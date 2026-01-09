//! Dimension type for physical dimensions.
//!
//! A dimension is represented as a 7-tuple of rational exponents for the
//! SI base dimensions: Length (L), Mass (M), Time (T), Current (I),
//! Temperature (Θ), Amount (N), and Luminous Intensity (J).

use num_rational::Rational32;
use pyo3::prelude::*;
use std::fmt;

/// Physical dimension as 7 rational exponents.
///
/// Represents the dimensional formula of a physical quantity using the
/// SI base dimensions:
/// - L: Length (meter)
/// - M: Mass (kilogram)
/// - T: Time (second)
/// - I: Electric current (ampere)
/// - Θ: Temperature (kelvin)
/// - N: Amount of substance (mole)
/// - J: Luminous intensity (candela)
#[pyclass]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RustDimension {
    /// Length exponent (L)
    #[pyo3(get)]
    pub length: (i32, i32),
    /// Mass exponent (M)
    #[pyo3(get)]
    pub mass: (i32, i32),
    /// Time exponent (T)
    #[pyo3(get)]
    pub time: (i32, i32),
    /// Current exponent (I)
    #[pyo3(get)]
    pub current: (i32, i32),
    /// Temperature exponent (Θ)
    #[pyo3(get)]
    pub temperature: (i32, i32),
    /// Amount exponent (N)
    #[pyo3(get)]
    pub amount: (i32, i32),
    /// Luminous intensity exponent (J)
    #[pyo3(get)]
    pub luminosity: (i32, i32),
}

impl RustDimension {
    /// Create a new dimension from rational exponents (as (numerator, denominator) pairs).
    pub fn new(
        length: (i32, i32),
        mass: (i32, i32),
        time: (i32, i32),
        current: (i32, i32),
        temperature: (i32, i32),
        amount: (i32, i32),
        luminosity: (i32, i32),
    ) -> Self {
        Self {
            length: Self::simplify(length),
            mass: Self::simplify(mass),
            time: Self::simplify(time),
            current: Self::simplify(current),
            temperature: Self::simplify(temperature),
            amount: Self::simplify(amount),
            luminosity: Self::simplify(luminosity),
        }
    }

    /// Simplify a fraction to lowest terms.
    fn simplify(frac: (i32, i32)) -> (i32, i32) {
        let r = Rational32::new(frac.0, frac.1);
        (*r.numer(), *r.denom())
    }

    /// Check if this dimension is compatible with another (for addition/subtraction).
    pub fn is_compatible(&self, other: &RustDimension) -> bool {
        self.length == other.length
            && self.mass == other.mass
            && self.time == other.time
            && self.current == other.current
            && self.temperature == other.temperature
            && self.amount == other.amount
            && self.luminosity == other.luminosity
    }

    /// Check if this is a dimensionless quantity.
    pub fn is_dimensionless(&self) -> bool {
        let zero = (0, 1);
        self.length == zero
            && self.mass == zero
            && self.time == zero
            && self.current == zero
            && self.temperature == zero
            && self.amount == zero
            && self.luminosity == zero
    }

    /// Multiply two dimensions (add exponents).
    pub fn multiply(&self, other: &RustDimension) -> RustDimension {
        RustDimension::new(
            Self::add_frac(self.length, other.length),
            Self::add_frac(self.mass, other.mass),
            Self::add_frac(self.time, other.time),
            Self::add_frac(self.current, other.current),
            Self::add_frac(self.temperature, other.temperature),
            Self::add_frac(self.amount, other.amount),
            Self::add_frac(self.luminosity, other.luminosity),
        )
    }

    /// Divide two dimensions (subtract exponents).
    pub fn divide(&self, other: &RustDimension) -> RustDimension {
        RustDimension::new(
            Self::sub_frac(self.length, other.length),
            Self::sub_frac(self.mass, other.mass),
            Self::sub_frac(self.time, other.time),
            Self::sub_frac(self.current, other.current),
            Self::sub_frac(self.temperature, other.temperature),
            Self::sub_frac(self.amount, other.amount),
            Self::sub_frac(self.luminosity, other.luminosity),
        )
    }

    /// Raise dimension to an integer power (scale exponents).
    pub fn power(&self, n: i32) -> RustDimension {
        RustDimension::new(
            Self::mul_frac(self.length, n),
            Self::mul_frac(self.mass, n),
            Self::mul_frac(self.time, n),
            Self::mul_frac(self.current, n),
            Self::mul_frac(self.temperature, n),
            Self::mul_frac(self.amount, n),
            Self::mul_frac(self.luminosity, n),
        )
    }

    /// Add two fractions.
    fn add_frac(a: (i32, i32), b: (i32, i32)) -> (i32, i32) {
        let ra = Rational32::new(a.0, a.1);
        let rb = Rational32::new(b.0, b.1);
        let result = ra + rb;
        (*result.numer(), *result.denom())
    }

    /// Subtract two fractions.
    fn sub_frac(a: (i32, i32), b: (i32, i32)) -> (i32, i32) {
        let ra = Rational32::new(a.0, a.1);
        let rb = Rational32::new(b.0, b.1);
        let result = ra - rb;
        (*result.numer(), *result.denom())
    }

    /// Multiply a fraction by an integer.
    fn mul_frac(a: (i32, i32), n: i32) -> (i32, i32) {
        let ra = Rational32::new(a.0, a.1);
        let result = ra * n;
        (*result.numer(), *result.denom())
    }
}

#[pymethods]
impl RustDimension {
    /// Create a new dimension from integer exponents.
    #[new]
    fn py_new(
        length: Option<i32>,
        mass: Option<i32>,
        time: Option<i32>,
        current: Option<i32>,
        temperature: Option<i32>,
        amount: Option<i32>,
        luminosity: Option<i32>,
    ) -> Self {
        RustDimension::new(
            (length.unwrap_or(0), 1),
            (mass.unwrap_or(0), 1),
            (time.unwrap_or(0), 1),
            (current.unwrap_or(0), 1),
            (temperature.unwrap_or(0), 1),
            (amount.unwrap_or(0), 1),
            (luminosity.unwrap_or(0), 1),
        )
    }

    /// Create a dimensionless dimension.
    #[staticmethod]
    fn dimensionless() -> Self {
        RustDimension::new((0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1))
    }

    /// Check if dimensions are compatible (for Python).
    fn is_compatible_py(&self, other: &RustDimension) -> bool {
        self.is_compatible(other)
    }

    /// Check if dimensionless (for Python).
    fn is_dimensionless_py(&self) -> bool {
        self.is_dimensionless()
    }

    /// Multiply dimensions (for Python).
    fn multiply_py(&self, other: &RustDimension) -> RustDimension {
        self.multiply(other)
    }

    /// Divide dimensions (for Python).
    fn divide_py(&self, other: &RustDimension) -> RustDimension {
        self.divide(other)
    }

    /// Power (for Python).
    fn power_py(&self, n: i32) -> RustDimension {
        self.power(n)
    }

    fn __repr__(&self) -> String {
        format!("{}", self)
    }

    fn __str__(&self) -> String {
        format!("{}", self)
    }

    fn __eq__(&self, other: &RustDimension) -> bool {
        self == other
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }
}

impl fmt::Display for RustDimension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts = Vec::new();

        let dims = [
            ("L", self.length),
            ("M", self.mass),
            ("T", self.time),
            ("I", self.current),
            ("Θ", self.temperature),
            ("N", self.amount),
            ("J", self.luminosity),
        ];

        for (name, (num, den)) in dims {
            if num != 0 {
                if den == 1 {
                    if num == 1 {
                        parts.push(name.to_string());
                    } else {
                        parts.push(format!("{}^{}", name, num));
                    }
                } else {
                    parts.push(format!("{}^({}/{})", name, num, den));
                }
            }
        }

        if parts.is_empty() {
            write!(f, "dimensionless")
        } else {
            write!(f, "{}", parts.join("·"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dimensionless() {
        let d = RustDimension::dimensionless();
        assert!(d.is_dimensionless());
    }

    #[test]
    fn test_length() {
        let d = RustDimension::new((1, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1));
        assert!(!d.is_dimensionless());
        assert_eq!(d.length, (1, 1));
    }

    #[test]
    fn test_multiply() {
        // m * s = m·s
        let m = RustDimension::new((1, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1));
        let s = RustDimension::new((0, 1), (0, 1), (1, 1), (0, 1), (0, 1), (0, 1), (0, 1));
        let ms = m.multiply(&s);
        assert_eq!(ms.length, (1, 1));
        assert_eq!(ms.time, (1, 1));
    }

    #[test]
    fn test_divide() {
        // m / s = m/s
        let m = RustDimension::new((1, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1));
        let s = RustDimension::new((0, 1), (0, 1), (1, 1), (0, 1), (0, 1), (0, 1), (0, 1));
        let mps = m.divide(&s);
        assert_eq!(mps.length, (1, 1));
        assert_eq!(mps.time, (-1, 1));
    }

    #[test]
    fn test_power() {
        // m^2
        let m = RustDimension::new((1, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1));
        let m2 = m.power(2);
        assert_eq!(m2.length, (2, 1));
    }
}
