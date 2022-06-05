//! sort along the axis and give the indices of those sorted values
//!

use ndarray::{Array, ArrayBase, Axis, Data, Dimension, RemoveAxis, Zip};
use num_traits::Float;
use ordered_float::OrderedFloat;

/// sort a float array along an axis and return the indices
pub fn argsortf<T, S, D>(a: &ArrayBase<S, D>, axis: usize) -> Array<usize, D>
where
    T: Float,
    S: Data<Elem = T>,
    D: Dimension + RemoveAxis,
{
    let mut output = Array::from_elem(a.dim(), 0);

    Zip::from(a.lanes(Axis(axis)))
        .and(output.lanes_mut(Axis(axis)))
        .for_each(|v, mut out| {
            let mut intermediate = v.iter().enumerate().collect::<Vec<_>>();
            intermediate.sort_by_key(|(_, f)| OrderedFloat(**f));
            let it = intermediate.into_iter().map(|(idx, _)| idx);

            let val = Array::from_iter(it);
            val.move_into(&mut out);
        });
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ndarray::array;

    #[test]
    fn test_argsortf() {
        let a = array![[25.6, 56.4, 35.6], [-45.5, 35.5, 25.8]];

        assert_eq!(argsortf(&a, 0), array![[1, 1, 1], [0, 0, 0]]);

        assert_eq!(argsortf(&a, 1), array![[0, 2, 1], [0, 2, 1]]);
    }
}
