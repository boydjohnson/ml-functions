//! Elementwise sigmoid ops
//!
//! ```rust
//!
//! use ml_functions::ops::Sigmoid;
//! use ml_functions::ndarray::Array;
//!
//! let arr = Array::from_shape_vec((2, 2), vec![0.0, 1.2, 0.0, -5.4]).unwrap();
//! let sigmoid = arr.sigmoid();
//!
//! ```

use crate::ops::Exp;
use ndarray::{Array, ArrayBase, Data, Dimension};
use num_traits::{Float, FromPrimitive};

/// Compute the elementwise sigmoid function
pub fn sigmoid<T, S, D>(a: &ArrayBase<S, D>) -> Array<T, D>
where
    T: Float + FromPrimitive,
    S: Data<Elem = T>,
    D: Dimension,
{
    a.map(|&v| -v)
        .exp()
        .map(|&v| T::one() + v)
        .map(|&denom| T::one() / denom)
}

/// Compute the elementwise sigmoid function
pub trait Sigmoid<T, D> {
    /// Compute the sigmoid function.
    fn sigmoid(&self) -> Array<T, D>;
}

impl<T, S, D> Sigmoid<T, D> for ArrayBase<S, D>
where
    T: Float + FromPrimitive,
    S: Data<Elem = T>,
    D: Dimension,
{
    fn sigmoid(&self) -> Array<T, D> {
        sigmoid(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ndarray::array;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_sigmoid() {
        let a = array![[1.456, 2.89105], [-2.567, 4.567]];

        assert_abs_diff_eq!(
            a.sigmoid(),
            array![
                [0.8109201227128108, 0.9474022289590442],
                [0.07129267889980415, 0.9897177429395645]
            ]
        );
    }
}
