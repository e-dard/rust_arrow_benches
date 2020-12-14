use std::arch::x86_64::*;

use arrow::{array, compute::kernels};

/// Filter and materialise functions are those that materialise a non-contiguous
/// sub-set of values in some array, which are defined by a filter (another
/// vector of indexes).
///
/// I care about the performance of these because in a columnar database you
/// often filter some column based on predicates applied to other columns and
/// are left with a set of indexes (`row_ids`) to materialise.
///
/// In my case at least it's OK to put a maximum row limit on a column of
/// u32::MAX so I use `u32` as row ids.

/// This is a relatively idiomatic Rust implementation of filter. It serves as a
/// baseline. I have arbitrarily picked 64-bit values since those are the most
/// common scalar types I deal with.
///
/// Also - use a pattern where a destination buffer is passed in, populated and
/// returned.
pub fn filter_materialise_values(values: &[u64], row_ids: &[u32], mut dst: Vec<u64>) -> Vec<u64> {
    dst.clear();
    dst.reserve(row_ids.len());

    for &id in row_ids.iter() {
        dst.push(values[id as usize]);
    }

    assert_eq!(dst.len(), row_ids.len());
    dst
}

/// This is an implementation of filter using Arrow arrays and kernels. Unlike
/// the basic rust version a reusable buffer is not passed in (though in
/// benchmarks) I'm not passing in a buffer to methods that could take one anyway.
pub fn filter_materialise_values_arrow(
    values: &array::UInt64Array,
    row_ids: &array::BooleanArray,
) -> std::sync::Arc<dyn arrow::array::Array> {
    kernels::filter::filter(values, row_ids).unwrap()
}

/// This is a more sophisticated implementation of filter using SIMD
/// intrinsics. I have arbitrarily picked 64-bit values since those are the most
/// common scalar types I deal with. In Rust it would not be a huge amount of
/// work to make this SIMD implementation generic (which is what Arrow does).
///
pub fn filter_materialise_values_simd(
    values: &[u64],
    row_ids: &[u32],
    mut dst: Vec<u64>,
) -> Vec<u64> {
    dst.clear();
    dst.reserve(row_ids.len());

    unsafe {
        let base_ptr = values.as_ptr() as *const i64;

        for chunk in row_ids.chunks_exact(4) {
            let chunk_ptr = chunk.as_ptr() as *const __m128i;
            let mat_values = _mm256_i32gather_epi64(base_ptr, _mm_loadu_si128(chunk_ptr), 8);

            _mm256_storeu_si256(dst.as_mut_ptr().add(dst.len()) as *mut __m256i, mat_values);
            dst.set_len(dst.len() + 4);
        }

        // materialise any remainder - maximum of three values. Not much value
        // in doing this in a SIMD register
        let rem = row_ids.len() - (row_ids.len() % 4);
        for &id in row_ids.iter().skip(rem) {
            dst.push(values[id as usize]);
        }
    }
    assert_eq!(dst.len(), row_ids.len());
    dst
}

mod test {

    #[test]
    fn filter_materialise_values() {
        assert_eq!(
            super::filter_materialise_values(
                (0..10).collect::<Vec<_>>().as_slice(),
                &[0, 1, 2, 3],
                vec![]
            ),
            vec![0_u64, 1, 2, 3]
        );
    }

    #[test]
    fn filter_materialise_values_arrow() {
        let values = arrow::array::UInt64Array::from((0..10).collect::<Vec<_>>());

        let mut filter = Vec::with_capacity(values.len());
        filter.resize(values.len(), false);
        for &i in [0_u32, 4, 5, 8].iter() {
            filter[i as usize] = true;
        }

        let row_ids = arrow::array::BooleanArray::from(filter);

        let exp = arrow::array::UInt64Array::from(vec![0, 4, 5, 8]);
        assert_eq!(
            super::filter_materialise_values_arrow(&values, &row_ids)
                .as_any()
                .downcast_ref::<arrow::array::UInt64Array>()
                .unwrap(),
            &exp
        );
    }

    #[test]
    fn filter_materialise_values_simd() {
        let cases = vec![
            (
                (100..110).collect::<Vec<_>>(),
                vec![0_u32, 1, 2, 3],
                vec![100_u64, 101, 102, 103],
            ),
            (
                (100..113).collect::<Vec<_>>(),
                vec![0, 12],
                vec![100_u64, 112],
            ),
            (
                vec![1020, u64::MAX, u64::MAX, u64::MAX, u64::MAX],
                vec![1, 2, 3, 4],
                vec![u64::MAX; 4],
            ),
            (
                vec![1020, u64::MAX, u64::MAX, u64::MAX, u64::MAX],
                vec![2],
                vec![u64::MAX],
            ),
            (
                vec![1020, u64::MAX, u64::MAX, u64::MAX, 29],
                vec![4],
                vec![29],
            ),
            (
                (100..1234).collect::<Vec<_>>(),
                (2..653).collect::<Vec<_>>(),
                (102..753).collect::<Vec<_>>(),
            ),
        ];

        for (values, row_ids, exp) in &cases {
            assert_eq!(
                &super::filter_materialise_values_simd(values, row_ids, vec![]),
                exp
            );
        }
    }
}
