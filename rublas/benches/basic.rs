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

fn basic_gemm_zero(crit: &mut Criterion) {
    let mut bench_group = crit.benchmark_group("basic_gemm_zero");
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

fn basic_gemm_uniform(crit: &mut Criterion) {
    let mut bench_group = crit.benchmark_group("basic_gemm_uniform");
    bench_group.sample_size(10);
    for Msize in vec![4096].iter() {
        for Ksize in vec![4096].iter() {
            bench_group.bench_with_input(
                BenchmarkId::new(format!("M{}_K{}_N{}_f32", Msize, Ksize, Msize), 0),
                Msize,
                |bench, msize| {
                    let lhs = Array2::<f32>::random([*Msize, *Ksize], Uniform::new(-1f32, 1.0));
                    let rhs = Array2::<f32>::random([*Ksize, *Msize], Uniform::new(-1f32, 1.0));
                    let mut out = Array2::<f32>::random([*Msize, *Msize], Uniform::new(-1f32, 1.0));
                    bench.iter(|| {
                        black_box(general_mat_mul(1.0, &lhs, &rhs, 1.0, &mut out));
                    });
                },
            );
        }
    }
}

criterion_group!(basic_tests, basic_gemm_zero, basic_gemm_uniform);
criterion_main!(basic_tests);
