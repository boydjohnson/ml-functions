//! argmax
//!
//! functionality for returning the indices of maximum values
//! along an axis.
//!

use ndarray::{Array, ArrayBase, Axis, Data, Dimension, RemoveAxis, Zip};
use num_traits::Float;
use ordered_float::OrderedFloat;

/// return indices of the maximum (Float) values along an axis.
pub fn argmaxf<T, S, D>(a: &ArrayBase<S, D>, axis: usize) -> Array<usize, <D as Dimension>::Smaller>
where
    T: Float,
    S: Data<Elem = T>,
    D: Dimension + RemoveAxis,
{
    Zip::from(a.lanes(Axis(axis))).map_collect(|v| {
        v.iter()
            .enumerate()
            .max_by_key(|(_, s)| OrderedFloat(**s))
            .map(|(idx, _)| idx)
            .expect("has at least one value")
    })
}

/// same as argmaxf but keep the same input dimensions.
pub fn argmaxf_keep_dims<T, S, D, Smaller>(a: &ArrayBase<S, D>, axis: usize) -> Array<usize, D>
where
    T: Float,
    S: Data<Elem = T>,
    D: Dimension<Smaller = Smaller> + RemoveAxis,
    Smaller: Dimension<Larger = D>,
{
    argmaxf(a, axis).insert_axis(Axis(axis))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ndarray::array;

    #[test]
    fn test_argmaxf() {
        let a = array![[3.4, 2.4, 10.4], [4.5, -1.5, 2.6], [2.4, 1.8, 8.9]];

        assert_eq!(argmaxf(&a, 0), array![1, 0, 0]);
        assert_eq!(argmaxf(&a, 1), array![2, 0, 2]);
    }

    #[test]
    fn test_argmaxf_keep_dims() {
        let a = array![[3.4, 2.4, 10.4], [4.5, -1.5, 2.6], [2.4, 1.8, 8.9]];

        assert_eq!(argmaxf_keep_dims(&a, 0), array![[1, 0, 0]]);
        assert_eq!(argmaxf_keep_dims(&a, 1), array![[2], [0], [2]]);
    }
}
