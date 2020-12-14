I work on a columnar database with a particular focus on in-memory execution of analytical-style queries.
Typically these often involve applying predicates (filtering) and aggregating or selecting row values.
Since these operations are almost always vectorisable there are a few technologies that are of particular interest in this area:

- LLVM's ability to auto-vectorise code in some cases;
- SIMD (in general): That is, the ability to write vectorised code manually with SIMD intrinsics;
- Arrow: A project that provides abstractions, APIs and kernels over immutable arrays of fixed-width or encoded data (also SIMD).

Currently I'm only thinking about performance on recent intel CPUs, e.g., those with `avx2` instructions.
I also typically don't need to think about columns with more values than can be expressed with 32-bit indexes.

There are a few "very hot operations" that happen on many queries, and often over large column sizes. For example:

##### Filter then materialise
You want to get a copy of some values from a column. These values are defined by a set of indexes in another array.

```
arr -> [10, 20, 30, 33, 40, 49]
filter -> [1, 2, 4]

select(arr, filter) -> [20, 30, 40]
```

##### Filter and aggregate
You want to apply some aggregate (or selector) to a set of values in a column.
These values are defined by a set of indexes in another array.

```
arr -> [10, 20, 30, 33, 40, 49]
filter -> [1, 2, 4]

sum(arr, filter) -> 90
```

```
arr -> [10, 8, 9, 33, 40, 49, 3]
filter -> [0, 2, 4, 5]

min(arr, filter) -> 9
```

## Current Benchmarks

This crate contains some benchmarks (which need DRYing up) that basically compare
three different implementations of some of these vectorised operations.
The implementations look like:

- Vanilla idiomatic Rust implementation: e.g., using a loop/iterator. LLVM may auto-vectorise some or all of it if we're lucky.
- Arrow Compute Kernel implementation: using some [Arrow compute kernels] implement the same operation.
- SIMD intrinsics: using [Intel's SIMD intrinsics] implement the same operation.

There are three different scenarios that are benchmarked right now:

- Filtering a column of values and materialising the results;
- Calculating a sum on a filtered set of values in a column; and
- Finding the max value on a filtered set of values in a column.

For each scenario, the three implementations are evaluated against five different 
filter inputs on a column of a million `u64` values.

- "uniform_density_10%": a filter that selects ~10% of the column from a uniform distribution, e.g., `[10, 14, 33, 88 ....]`.
- "uniform_density_50%": as above but selects ~50% of the column.
- "uniform_density_75%": as above but selects ~75% of the column.
- "uniform_density_5%_block_size_5": a filter that select ~5% of the column values but then select a run of 5 subsequent values, e.g., `[17, 18, 19, 20, 21, 87, 88, 89, 90, 91...]`. This closely mimics the shape of data in columns that have been sorted by other columns first.
- "uniform_density_10%_block_size_10": as above but a run of 10 values each time a value is selected.

Therefore in total there are 45 benchmarks here:


[Arrow compute kernels]: https://docs.rs/arrow/2.0.0/arrow/compute/kernels/index.html
[Intel's SIMD intrinsics]: https://software.intel.com/sites/landingpage/IntrinsicsGuide/

## Current Results

As of Arrow commit `5353c285c6dfb3381ac0f1c9e7cd63d7fcb8da4a` the results are below.
Benchmarks were ran on a 2020 i9 MacBook Pro with the following `cargo` invocation:

```shell
$ RUSTFLAGS="-C target-cpu=native -C target-feature=+avx2" cargo bench
```

In each table the results are displayed in "Melem/s", which means "millions of elements per second" (to go to `GB/s` multiply by `8` ). Bigger is better.

### Filter Values

| Implementation  |  uniform_10% | uniform_50% | uniform_75% | uniform_5%_bs_5 | uniform_10%_bs_10
| -------------:   | :-------------: |:-------------: | :-------------: | :-------------: | :-------------: |
| Vanilla (safe) Rust  | 519 Melem/s  | 504  | 448  | 482  | 492  |
| Arrow Compute Kernels  | 59.5   | 87.8  | 146  | 113  | 207 |
| SIMD intrinsics (unsafe) Rust  |  888   | 759  | 671  | 966  | 682  |

### Filter Max

| Implementation  |  uniform_10% | uniform_50% | uniform_75% | uniform_5%_bs_5 | uniform_10%_bs_10
| -------------:   | :-------------: |:-------------: | :-------------: | :-------------: | :-------------: |
| Vanilla (safe) Rust  | 751   | 1016  | 1147  | 1100  | 1174  |
| Arrow Compute Kernels  | 59   | 85  | 140  | 117  | 197  |
| SIMD intrinsics (unsafe) Rust  | 1016   | 1871  | 1950  | 1619  | 1945  |

### Filter Sum

| Implementation  |  uniform_10% | uniform_50% | uniform_75% | uniform_5%_bs_5 | uniform_10%_bs_10
| -------------:   | :-------------: |:-------------: | :-------------: | :-------------: | :-------------: |
| Vanilla (safe) Rust  | 919   | 1660  | 1838  | 1389  | 1819  |
| Arrow Compute Kernels  | 59   | 78  | 135  | 114  | 178  |
| SIMD intrinsics (unsafe) Rust  | 1050   | 2033  | 2366  | 1769  | 2326  |

## Summary

- Hey! The non-arrow code doesn't handle nulls at all. No it doesn't, but that would at most cost a 2x performance drop (less really), so even with null support there is a huge discrepancy.
- Hey! The sum/max SIMD doesn't handle ABC edge-case. Probably not, but the perf difference is so big the existing implementations are demonstrative enough.

Overall my current takeaway is that:

- "Vanilla" Rust is very fast out the box.
- But, you can still get a really significant boost from using SIMD enabled code.
- The Rust Arrow crate has some significant performance issues in it **but** at least from what I have seen the issues are higher up in the library than the SIMD computations.

I have profiled these implementations a bunch and I'm pretty sure that the Arrow crate can be significantly improved, which is a good thing because hand-writing SIMD is not a trivial thing to do, though for the right hot loop it really does give the best payoff.
