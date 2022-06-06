//! sort along an axis

use ndarray::{Array, ArrayBase, Axis, Data, Dimension, Zip};
use num_traits::Float;
use ordered_float::OrderedFloat;

/// sortf sort floats along axis returning an Array of same shape.
pub fn sortf<T, S, D>(a: &ArrayBase<S, D>, axis: usize) -> Array<T, D>
where
    T: Float,
    S: Data<Elem = T>,
    D: Dimension,
{
    let mut output = Array::from_elem(a.dim(), T::zero());

    Zip::from(a.lanes(Axis(axis)))
        .and(output.lanes_mut(Axis(axis)))
        .for_each(|v, mut out| {
            let mut intermediate = v.iter().collect::<Vec<_>>();
            intermediate.sort_by_key(|f| OrderedFloat(**f));
            let it = intermediate.into_iter().copied();

            let val = Array::from_iter(it);
            val.move_into(&mut out);
        });
    output
}

/// Sort floats along an axis
pub trait Sortf<T, D> {
    /// sort floats along an axis
    fn sortf(&self, axis: usize) -> Array<T, D>;
}

impl<T, S, D> Sortf<T, D> for ArrayBase<S, D>
where
    T: Float,
    S: Data<Elem = T>,
    D: Dimension,
{
    fn sortf(&self, axis: usize) -> Array<T, D> {
        sortf(self, axis)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ndarray::array;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_sortf() {
        let a = array![[25.6, 56.4, 35.6], [-45.5, 35.5, 25.8]];

        assert_abs_diff_eq!(
            sortf(&a, 0),
            array![[-45.5, 35.5, 25.8], [25.6, 56.4, 35.6]]
        );

        assert_abs_diff_eq!(
            sortf(&a, 1),
            array![[25.6, 35.6, 56.4], [-45.5, 25.8, 35.5]]
        );
    }
}
