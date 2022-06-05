//! softmax creates values that sum to 1, along an axis.
//!
//! ```rust
//! use ml_functions::ops::softmax;
//! use ml_functions::ndarray::Array;
//!
//! let a = Array::from_shape_vec((2, 2), vec![-0.5, 0.4, 0.7, -0.056]).unwrap();
//! softmax(&a, 1);
//! ```
//!
//!

use crate::ops::{exp, MaxfKeepDims};
use ndarray::{Array, ArrayBase, Axis, Data, Dimension, RemoveAxis};
use num_traits::Float;
use std::ops::{Div, Sub};

/// the softmax function.
pub fn softmax<T, S, D, Smaller>(a: &ArrayBase<S, D>, axis: usize) -> Array<T, D>
where
    T: Float,
    S: Data<Elem = T>,
    D: Dimension<Smaller = Smaller> + RemoveAxis,
    Smaller: Dimension<Larger = D>,
{
    let max = a.maxf_keep_dims(axis);

    let exp_shifted = exp(&a.sub(max));

    let divisor = exp_shifted.sum_axis(Axis(axis)).insert_axis(Axis(axis));

    exp_shifted.div(divisor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ndarray::array;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_softmax() {
        let a = array![[-0.5, 0.4], [0.7, -0.056]];
        assert_abs_diff_eq!(
            softmax(&a, 1),
            array![
                [0.289050497374996, 0.710949502625004],
                [0.680484661216602, 0.319515338783398]
            ]
        );

        assert_abs_diff_eq!(
            softmax(&a, 0),
            array![
                [0.23147521650098232, 0.6120648370660614],
                [0.7685247834990175, 0.38793516293393854]
            ]
        );

        let a = array![[-4.555556, 1.23, 0.5554], [3.2222, -4.5, 1.2222]];

        assert_abs_diff_eq!(
            softmax(&a, 0),
            array![
                [
                    4.1877580053227700e-04,
                    9.9676343215216312e-01,
                    3.3921374422284389e-01
                ],
                [
                    9.9958122419946793e-01,
                    3.2365678478368408e-03,
                    6.6078625577715611e-01
                ]
            ]
        );

        assert_abs_diff_eq!(
            softmax(&a, 1),
            array![
                [
                    2.0309029739572094e-03,
                    6.6118687065770032e-01,
                    3.3678222636834249e-01
                ],
                [
                    8.8045362008554817e-01,
                    3.8993986347313859e-04,
                    1.1915644005097865e-01
                ]
            ]
        )
    }
}
