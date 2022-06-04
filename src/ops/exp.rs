//! Exponential function
//!
//! ```rust
//!
//! use ml_functions::{ops::Exp, ndarray::Array};
//!
//! let arr = Array::from_shape_vec((2, 2), vec![-0.1, 0.0, 0.0, 0.1]).unwrap();
//! let exponential = arr.exp();
//!
//! ```

use ndarray::{Array, ArrayBase, Data, Dimension};
use num_traits::Float;

/// The exponential function.
pub fn exp<T, S, D>(x: &ArrayBase<S, D>) -> Array<T, D>
where
    T: Float,
    S: Data<Elem = T>,
    D: Dimension,
{
    x.map(|v| v.exp())
}

/// The exponential function.
pub trait Exp<T, D> {
    /// Compute the element-wise exponental function.
    fn exp(&self) -> Array<T, D>;
}

impl<T, S, D> Exp<T, D> for ArrayBase<S, D>
where
    T: Float,
    S: Data<Elem = T>,
    D: Dimension,
{
    fn exp(&self) -> Array<T, D> {
        exp(self)
    }
}
