use std::arch::x86_64::*;

use arrow::{array, compute::kernels};

/// Filter and aggregate functions are those that aggregate over a
/// non-contiguous sub-set of values in some array, where the set of values to
/// aggregate is defined by a filter (another vector of indexes).
///
/// I care about the performance of these because in a columnar database you
/// often need to do some vectorised max selector based on row ids calculated
/// from applying predicates to other columns.
///
/// In my case at least it's OK to put a maximum row limit on a column of
/// u32::MAX so I use `u32` as row ids.

/// This is a relatively idiomatic Rust implementation of filter_min. It serves
/// as a baseline. I have arbitrarily picked 64-bit values since those are the
/// most common scalar types I deal with.
///
pub fn filter_max(values: &[u64], row_ids: &[u32]) -> u64 {
    row_ids.iter().map(|&id| values[id as usize]).max().unwrap()
}

/// This is an implementation of filter and max using Arrow arrays and kernels.
/// Currently Arrow needs to perform this aggregation operation as two steps
/// (filter then max).
pub fn filter_max_arrow(values: &array::UInt64Array, row_ids: &array::BooleanArray) -> u64 {
    let filter_result = kernels::filter::filter(values, row_ids).unwrap();
    kernels::aggregate::max(
        filter_result
            .as_any()
            .downcast_ref::<arrow::array::UInt64Array>()
            .unwrap(),
    )
    .unwrap()
}

/// This is an implementation of filter then max using SIMD intrinsics. I have
/// picked 64-bit values since those are the most common scalar types I deal
/// with. In Rust it would not be a huge amount of work to make this SIMD
/// implementation generic (which is what Arrow does).
///
/// NOTE!!! This implementation is not correct for large unsigned values. The
/// SIMD intrinsics work on signed integers. Once you set the high bit on an
/// unsigned value it will be treated as a negative number.
///
/// One way around that might be to unset the high bit on all values to be
/// compared and somehow set it back after. Need to think about that and/or do
/// some reading.
///
pub fn filter_max_simd(values: &[u64], row_ids: &[u32]) -> u64 {
    if row_ids.len() < 4 {
        return filter_max(values, row_ids);
    }

    unsafe {
        let base_ptr = values.as_ptr() as *const i64;

        let mut max_lanes = _mm256_i32gather_epi64(
            base_ptr,
            _mm_loadu_si128(row_ids.as_ptr() as *const __m128i),
            8,
        );

        for chunk in row_ids.chunks_exact(4).skip(1) {
            let chunk_ptr = chunk.as_ptr() as *const __m128i;
            let row_values = _mm256_i32gather_epi64(base_ptr, _mm_loadu_si128(chunk_ptr), 8);

            let max_mask = _mm256_cmpgt_epi64(row_values, max_lanes);
            max_lanes = _mm256_blendv_epi8(max_lanes, row_values, max_mask);
        }

        let result: [u64; 4] = std::mem::transmute(max_lanes);

        // find the max in any remainder - at most three values. Not much value
        // in doing this in a SIMD register
        let rem = row_ids.len() - (row_ids.len() % 4);
        let rem_max = row_ids
            .iter()
            .skip(rem)
            .map(|&id| values[id as usize])
            .max();

        match rem_max {
            Some(rm) => rm.max(*result.iter().max().unwrap()),
            None => *result.iter().max().unwrap(),
        }
    }
}

mod test {

    #[test]
    fn filter_max() {
        assert_eq!(
            super::filter_max((12..39).collect::<Vec<_>>().as_slice(), &[0, 1, 2, 6, 8]),
            20
        );
    }

    #[test]
    fn filter_max_arrow() {
        let values = arrow::array::UInt64Array::from((12..378).collect::<Vec<_>>());

        let mut filter = Vec::with_capacity(values.len());
        filter.resize(values.len(), false);
        for &i in [16, 22, 23].iter() {
            filter[i as usize] = true;
        }

        let row_ids = arrow::array::BooleanArray::from(filter);

        assert_eq!(super::filter_max_arrow(&values, &row_ids), 35);
    }

    fn sum_slice(values: &[u64]) -> u64 {
        values.iter().sum()
    }

    #[test]
    fn filter_max_simd() {
        let cases = vec![
            ((100..110).collect::<Vec<_>>(), vec![0_u32, 1, 2, 3], 103),
            ((100..113).collect::<Vec<_>>(), vec![0, 12], 112),
            (vec![20], vec![0_u32], 20),
            (vec![20, 10, 20, 3], vec![1, 3], 10),
            (
                vec![1020, 1023, 100, 3498, u32::MAX as u64],
                vec![0, 1, 2, 3, 4],
                u32::MAX as u64,
            ),
            (
                vec![1021, 1023, 100, 3498, u32::MAX as u64, 1020],
                vec![3, 4],
                u32::MAX as u64,
            ),
            (
                (100..1234).collect::<Vec<_>>(),
                vec![3, 2, 5, 10, 10, 11, 21],
                121,
            ),
            (
                vec![
                    21915, 99007, 8047, 46274, 90428, 11590, 24439, 44017, 80634, 73623, 28791,
                    34440, 35442, 70, 53834, 19529, 74056, 6737, 42825, 4378, 78251, 39440, 45815,
                    199, 200,
                ],
                vec![11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24],
                78251,
            ),
        ];

        for (values, row_ids, exp) in &cases {
            assert_eq!(&super::filter_max_simd(values, row_ids), exp);
        }
    }
}
