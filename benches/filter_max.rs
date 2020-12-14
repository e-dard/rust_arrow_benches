use std::fmt;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::{distributions, rngs::ThreadRng, Rng};

use rust_arrow_benches::filter_max;

const ROWS: usize = 1_000_003; // ~1 million values in the column for now. (3 encourages non-chunking edge cases)

enum FilterType {
    // a filter with uniformly distributed rows of a certain density
    // (10 would be 10% of rows)
    Uniform(Vec<u32>, usize),

    // a filter with a run of rows distributed through a column. This more closely
    // mimics a column that has been sorted by some other columns.
    Run(Vec<u32>, usize, usize),
}

impl FilterType {
    fn len(&self) -> usize {
        match self {
            FilterType::Uniform(v, _) => v.len(),
            FilterType::Run(v, _, _) => v.len(),
        }
    }

    fn as_slice(&self) -> &[u32] {
        match self {
            FilterType::Uniform(v, _) => v.as_slice(),
            FilterType::Run(v, _, _) => v.as_slice(),
        }
    }
}

impl fmt::Display for FilterType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FilterType::Uniform(_, density) => write!(f, "uniform_density_{:?}%", density),
            FilterType::Run(_, density, block_size) => write!(
                f,
                "uniform_density_{:?}%_block_size_{:?}",
                density, block_size
            ),
        }
    }
}

// Create a set of row_ids to apply to a column. Provide a prng, the domain that
// the row_ids can be picked from (`n`) and the probability of a row being
// selected, represented as `1/prop`.
fn random_filter(rng: &mut ThreadRng, n: usize, prop: usize) -> Vec<u32> {
    let dist = distributions::Uniform::from(0..100);
    rng.sample_iter(dist)
        .enumerate()
        .take(n)
        .filter_map(|(row_id, x)| {
            if x < prop {
                return Some(row_id as u32);
            }
            None
        })
        .collect::<Vec<_>>()
}

// Create a set of row_ids to apply to a column using a strategy where "runs"
// of matching rows are created according to 1/prop probability.
fn random_filter_run(rng: &mut ThreadRng, n: usize, prop: usize, run_size: usize) -> Vec<u32> {
    let dist = distributions::Uniform::from(0..100);

    // this is not at all perfect. When the prng decides to emit a run
    // of row ids it doesn't skip the `for` to the end of the run, which means
    // you can lead to larger blocks than `run_size`. The general data layout
    // is okay though for the use-case.
    let mut result = vec![];
    for row_id in 0..n {
        if rng.sample(dist) < prop {
            result.extend(row_id..row_id + run_size);
        }
    }

    // This generator is a bit ghetto - it could generate row_ids that are
    // upto block_size-1 over the max. It can also generate duplicates so remove
    // those.
    result
        .into_iter()
        .filter_map(|row_id| {
            if row_id < n - 1 {
                Some(row_id as u32)
            } else {
                None
            }
        })
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn bench_filter_max(c: &mut Criterion) {
    let mut rng = rand::thread_rng();

    // initialise column with random values.
    let col = rng
        .sample_iter(distributions::Uniform::from(0..100000))
        .take(ROWS)
        .collect::<Vec<_>>();

    // initialise different filters on the above column (create a set of row_ids to apply to col)
    let filter_types = vec![
        FilterType::Uniform(random_filter(&mut rng, ROWS, 10), 10),
        FilterType::Uniform(random_filter(&mut rng, ROWS, 50), 50),
        FilterType::Uniform(random_filter(&mut rng, ROWS, 75), 75),
        FilterType::Run(random_filter_run(&mut rng, ROWS, 5, 5), 5, 5),
        FilterType::Run(random_filter_run(&mut rng, ROWS, 10, 10), 10, 10),
    ];

    for filter_type in &filter_types {
        filter_max_rust_idiomatic(c, &col, filter_type);
        filter_max_arrow(c, &col, filter_type);
        filter_max_simd(c, &col, filter_type);
    }
}

fn filter_max_rust_idiomatic(c: &mut Criterion, col: &[u64], row_ids: &FilterType) {
    let mut group = c.benchmark_group("filter_max_rust_idiomatic");

    group.throughput(Throughput::Elements(row_ids.len() as u64));
    group.bench_function(BenchmarkId::from_parameter(format!("{}", row_ids)), |b| {
        b.iter(|| {
            let result = filter_max::filter_max(col, row_ids.as_slice());
            assert!(result > 0); // ensure bench doesn't get optimised away
        });
    });
}

fn filter_max_arrow(c: &mut Criterion, col: &[u64], row_ids: &FilterType) {
    let mut group = c.benchmark_group("filter_max_arrow");

    // for assertion
    let max = filter_max::filter_max(&col, row_ids.as_slice());

    group.throughput(Throughput::Elements(row_ids.len() as u64));

    let col_arr = arrow::array::UInt64Array::from(col.to_owned());
    let mut filter = Vec::with_capacity(col_arr.len());
    filter.resize(col_arr.len(), false);
    for &row_id in row_ids.as_slice().iter() {
        filter[row_id as usize] = true;
    }
    let row_ids_arr = arrow::array::BooleanArray::from(filter);

    group.bench_function(BenchmarkId::from_parameter(format!("{}", row_ids)), |b| {
        b.iter(|| {
            let result = filter_max::filter_max_arrow(&col_arr, &row_ids_arr);
            assert_eq!(result, max); // ensure bench not optimised away
        });
    });
}

fn filter_max_simd(c: &mut Criterion, col: &[u64], row_ids: &FilterType) {
    let mut group = c.benchmark_group("filter_max_simd");

    // for assertion
    let max = filter_max::filter_max(&col, row_ids.as_slice());
    group.throughput(Throughput::Elements(row_ids.len() as u64));
    group.bench_function(BenchmarkId::from_parameter(format!("{}", row_ids)), |b| {
        b.iter(|| {
            let result = filter_max::filter_max_simd(col, row_ids.as_slice());
            assert_eq!(result, max);
        });
    });
}

criterion_group!(benches, bench_filter_max);
criterion_main!(benches);
