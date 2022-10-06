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
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use ndarray_rand::F32;

use criterion::*;
use rublas::prelude::*;

// TODO add a gemv_legacy
fn gemm_ndarray(crit: &mut Criterion) {
    let mut bench_group = crit.benchmark_group("gemm_ndarray");
    bench_group.sample_size(10);
    for Msize in vec![16, 256, 4096].iter() {
        for Ksize in vec![16, 256, 4096].iter() {
            bench_group.bench_with_input(
                BenchmarkId::new(format!("M{}_K{}_f32", Msize, Ksize), 0),
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
            bench_group.bench_with_input(
                BenchmarkId::new(format!("M{}_K{}_f64", Msize, Ksize), 0),
                Msize,
                |bench, msize| {
                    let a = Array2::<f64>::zeros((*Msize, *Ksize));
                    let (m, n) = a.dim();
                    let x = Array1::<f64>::zeros(n);
                    let mut y = Array1::<f64>::zeros(m);
                    bench.iter(|| {
                        black_box(general_mat_vec_mul(1.0, &a, &x, 1.0, &mut y));
                    });
                },
            );
        }
    }
}

fn gemm_rublas(crit: &mut Criterion) {
    let mut bench_group = crit.benchmark_group("gemm_rublas");
    bench_group.sample_size(10);
    for Msize in vec![16, 256, 4096].iter() {
        for Ksize in vec![16, 256, 4096].iter() {
            bench_group.bench_with_input(
                BenchmarkId::new(format!("M{}_K{}_f32", Msize, Ksize), 0),
                Msize,
                |bench, msize| {
                    let lhs = BlasTensor::ones(vec![*Msize, *Ksize]);
                    let rhs = BlasTensor::ones(vec![*Ksize, *Msize]);
                    let mut out = BlasTensor::zeros(vec![*Msize, *Msize]);
                    let exec = BlasExecutor::new();
                    bench.iter(|| {
                        black_box(exec.gemm_nn(&lhs, &rhs, &mut out));
                    });
                },
            );
        }
    }
}

criterion_group!(gemm_tests, gemm_ndarray, gemm_rublas);
criterion_main!(gemm_tests);
