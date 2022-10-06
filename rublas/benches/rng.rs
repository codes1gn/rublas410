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

fn rng_zeros(crit: &mut Criterion) {
    let mut bench_group = crit.benchmark_group("rng_zeros");
    for mat_size in vec![16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192].iter() {
        bench_group.bench_with_input(
            BenchmarkId::new(format!("{}_{}_f32", mat_size, mat_size), mat_size),
            mat_size,
            |bench, msize| {
                bench.iter(|| {
                    black_box(Array::<f32, _>::zeros((*msize, *msize)));
                });
            },
        );
        bench_group.bench_with_input(
            BenchmarkId::new(format!("{}_{}_f64", mat_size, mat_size), mat_size),
            mat_size,
            |bench, msize| {
                bench.iter(|| {
                    black_box(Array::<f32, _>::zeros((*msize, *msize)));
                });
            },
        );
    }
}

fn rng_uniform(crit: &mut Criterion) {
    let mut bench_group = crit.benchmark_group("rng_uniform");
    for mat_size in vec![16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192].iter() {
        bench_group.bench_with_input(
            BenchmarkId::new(format!("{}_{}_f32", mat_size, mat_size), mat_size),
            mat_size,
            |bench, msize| {
                bench.iter(|| {
                    black_box(Array::random((*msize, *msize), Uniform::new(-1f32, 1.)));
                });
            },
        );
        bench_group.bench_with_input(
            BenchmarkId::new(format!("{}_{}_f64", mat_size, mat_size), mat_size),
            mat_size,
            |bench, msize| {
                bench.iter(|| {
                    black_box(Array::random((*msize, *msize), Uniform::new(-1f64, 1.)));
                });
            },
        );
    }
}

fn rng_normal(crit: &mut Criterion) {
    let mut bench_group = crit.benchmark_group("rng_normal");
    for mat_size in vec![16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192].iter() {
        bench_group.bench_with_input(
            BenchmarkId::new(format!("{}_{}_f32", mat_size, mat_size), mat_size),
            mat_size,
            |bench, msize| {
                bench.iter(|| {
                    black_box(Array::random(
                        (*msize, *msize),
                        Normal::new(-1f32, 1.).unwrap(),
                    ));
                });
            },
        );
        bench_group.bench_with_input(
            BenchmarkId::new(format!("{}_{}_f64", mat_size, mat_size), mat_size),
            mat_size,
            |bench, msize| {
                bench.iter(|| {
                    black_box(Array::random(
                        (*msize, *msize),
                        Normal::new(-1f64, 1.).unwrap(),
                    ));
                });
            },
        );
    }
}

// TODO other rng patterns
criterion_group!(rng_tests, rng_zeros, rng_uniform, rng_normal);
criterion_main!(rng_tests);

//  #[bench]
//  fn norm_f32(b: &mut Bencher) {
//      let m = 100;
//      b.iter(|| Array::random((m, m), F32(Normal::new(0., 1.).unwrap())));
//  }
//
//  #[bench]
//  fn norm_f64(b: &mut Bencher) {
//      let m = 100;
//      b.iter(|| Array::random((m, m), Normal::new(0., 1.).unwrap()));
//  }
