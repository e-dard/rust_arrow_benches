use std::arch::x86_64::*;

use arrow::{array, compute::kernels};

/// Filter and aggregate functions are those that aggregate over a
/// non-contiguous sub-set of values in some array, where the set of values to
/// aggregate is defined by a filter (another vector of indexes).
///
/// I care about the performance of these because in a columnar database you
/// often need to do some vectorised summation based on row ids calculated from
/// applying predicates to other columns.
///
/// In my case at least it's OK to put a maximum row limit on a column of
/// u32::MAX so I use `u32` as row ids.

///
/// *Note* - these implementations all barf in the same way on overflow, so in
/// that sense they're basically doing the same thing.
///

/// This is a relatively idiomatic Rust implementation of filter_sum. It serves
/// as a baseline. I have arbitrarily picked 64-bit values since those are the
/// most common scalar types I deal with.
///
pub fn filter_sum(values: &[u64], row_ids: &[u32]) -> u64 {
    let mut result = 0;
    for &id in row_ids.iter() {
        result += values[id as usize];
    }
    result
}

/// This is an implementation of filter and sum using Arrow arrays and kernels.
/// Currently Arrow needs to perform this aggregation operation as two steps
/// (filter then sum).
pub fn filter_sum_arrow(values: &array::UInt64Array, row_ids: &array::BooleanArray) -> u64 {
    let filter_result = kernels::filter::filter(values, row_ids).unwrap();
    kernels::aggregate::sum(
        filter_result
            .as_any()
            .downcast_ref::<arrow::array::UInt64Array>()
            .unwrap(),
    )
    .unwrap()
}

/// This is an implementation of filter then sum using SIMD intrinsics. I have
/// picked 64-bit values since those are the most common scalar types I deal
/// with. In Rust it would not be a huge amount of work to make this SIMD
/// implementation generic (which is what Arrow does).
///
pub fn filter_sum_simd(values: &[u64], row_ids: &[u32]) -> u64 {
    unsafe {
        let base_ptr = values.as_ptr() as *const i64;
        let mut sum_lanes = _mm256_setzero_si256(); // u64x4

        for chunk in row_ids.chunks_exact(4) {
            let chunk_ptr = chunk.as_ptr() as *const __m128i;
            let row_values = _mm256_i32gather_epi64(base_ptr, _mm_loadu_si128(chunk_ptr), 8);
            sum_lanes = _mm256_add_epi64(sum_lanes, row_values);
        }

        // sum any remainder - maximum of three values. Not much value
        // in doing this in a SIMD register
        let rem = row_ids.len() - (row_ids.len() % 4);
        let rem_sum = row_ids
            .iter()
            .skip(rem)
            .map(|&id| values[id as usize])
            .sum::<u64>();

        let result: (u64, u64, u64, u64) = std::mem::transmute(sum_lanes);
        result.0 + result.1 + result.2 + result.3 + rem_sum
    }
}

mod test {

    #[test]
    fn filter_sum() {
        assert_eq!(
            super::filter_sum((0..10).collect::<Vec<_>>().as_slice(), &[0, 1, 2, 3]),
            6
        );
    }

    #[test]
    fn filter_sum_arrow() {
        let values = arrow::array::UInt64Array::from((0..10).collect::<Vec<_>>());

        let mut filter = Vec::with_capacity(values.len());
        filter.resize(values.len(), false);
        for &i in [0_u32, 4, 5, 8].iter() {
            filter[i as usize] = true;
        }

        let row_ids = arrow::array::BooleanArray::from(filter);

        assert_eq!(super::filter_sum_arrow(&values, &row_ids), 17);
    }

    fn sum_slice(values: &[u64]) -> u64 {
        values.iter().sum()
    }

    #[test]
    fn filter_sum_values_simd() {
        let cases = vec![
            (
                (100..110).collect::<Vec<_>>(),
                vec![0_u32, 1, 2, 3],
                sum_slice(&[100_u64, 101, 102, 103]),
            ),
            (
                (100..113).collect::<Vec<_>>(),
                vec![0, 12],
                sum_slice(&[100_u64, 112]),
            ),
            (
                vec![1020, 1023, 100, 3498, u32::MAX as u64],
                vec![1, 2, 3, 4],
                sum_slice(&[1023, 100, 3498, u32::MAX as u64]),
            ),
            (
                (100..1234).collect::<Vec<_>>(),
                (2..653).collect::<Vec<_>>(),
                sum_slice(&(102..753).collect::<Vec<_>>()),
            ),
        ];

        for (values, row_ids, exp) in &cases {
            assert_eq!(&super::filter_sum_simd(values, row_ids), exp);
        }
    }

    #[test]
    #[should_panic]
    fn filter_sum_overflow() {
        super::filter_sum(vec![u64::MAX, 1].as_slice(), &[0, 1]);
    }

    #[test]
    #[should_panic]
    fn filter_sum_arrow_overflow() {
        let values = arrow::array::UInt64Array::from(vec![u64::MAX, 1]);
        let row_ids = arrow::array::BooleanArray::from(vec![true, true]);
        super::filter_sum_arrow(&values, &row_ids);
    }

    #[test]
    #[should_panic]
    fn filter_sum_simd_overflow() {
        super::filter_sum_simd(vec![u64::MAX, 1].as_slice(), &[0, 1]);
    }
}
