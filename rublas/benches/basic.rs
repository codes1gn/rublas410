#[cfg(feature = "openblas")]
extern crate ndarray_blas as ndarray;

#[cfg(feature = "netlib")]
extern crate ndarray_blas as ndarray;

#[cfg(all(
    feature = "default",
    not(feature = "openblas"),
    not(feature = "netlib")
))]
extern crate ndarray_vanilla as ndarray;

use ndarray::linalg::general_mat_mul;
use ndarray::linalg::general_mat_vec_mul;
use ndarray::prelude::*;
use ndarray::Array;
use ndarray::*;

use criterion::*;
use rublas::prelude::*;

fn basic_gemv(crit: &mut Criterion) {
    let mut bench_group = crit.benchmark_group("basic_gemv");
    bench_group.sample_size(10);
    for Msize in vec![8192].iter() {
        for Ksize in vec![8192].iter() {
            bench_group.bench_with_input(
                BenchmarkId::new(format!("M{}_K{}_f32", Msize, Ksize), 0),
                Msize,
                |bench, msize| {
                    let a = Array2::<f32>::zeros((*Msize, *Ksize));
                    let (m, n) = a.dim();
                    let x = Array1::<f32>::zeros(n);
                    let mut y = Array1::<f32>::zeros(m);
                    bench.iter(|| {
                        black_box(general_mat_vec_mul(1.0, &a, &x, 1.0, &mut y));
                    });
                },
            );
        }
    }
}

fn basic_gemm(crit: &mut Criterion) {
    let mut bench_group = crit.benchmark_group("basic_gemm");
    bench_group.sample_size(10);
    for Msize in vec![4096].iter() {
        for Ksize in vec![4096].iter() {
            bench_group.bench_with_input(
                BenchmarkId::new(format!("M{}_K{}_N{}_f32", Msize, Ksize, Msize), 0),
                Msize,
                |bench, msize| {
                    let lhs = Array2::<f32>::zeros((*Msize, *Ksize));
                    let rhs = Array2::<f32>::zeros((*Ksize, *Msize));
                    let mut out = Array2::<f32>::zeros((*Msize, *Msize));
                    bench.iter(|| {
                        black_box(general_mat_mul(1.0, &lhs, &rhs, 1.0, &mut out));
                    });
                },
            );
        }
    }
}

criterion_group!(basic_tests, basic_gemv, basic_gemm);
criterion_main!(basic_tests);
