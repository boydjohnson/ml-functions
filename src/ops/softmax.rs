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

/// the softmax function.
pub trait Softmax<T, D> {
    /// the softmax function.
    fn softmax(&self, axis: usize) -> Array<T, D>;
}

impl<T, S, D, Smaller> Softmax<T, D> for ArrayBase<S, D>
where
    T: Float,
    S: Data<Elem = T>,
    D: Dimension<Smaller = Smaller> + RemoveAxis,
    Smaller: Dimension<Larger = D>,
{
    fn softmax(&self, axis: usize) -> Array<T, D> {
        softmax(self, axis)
    }
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
                    4.187_758_005_322_77e-4,
                    9.967_634_321_521_631e-1,
                    3.392_137_442_228_439e-1
                ],
                [
                    9.995_812_241_994_679e-1,
                    3.236_567_847_836_840_8e-3,
                    6.607_862_557_771_561e-1
                ]
            ]
        );

        assert_abs_diff_eq!(
            softmax(&a, 1),
            array![
                [
                    2.030_902_973_957_209_4e-3,
                    6.611_868_706_577_003e-1,
                    3.367_822_263_683_425e-1
                ],
                [
                    8.804_536_200_855_482e-1,
                    3.899_398_634_731_386e-4,
                    1.191_564_400_509_786_5e-1
                ]
            ]
        )
    }
}
