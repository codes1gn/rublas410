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

use criterion::*;
use rublas::prelude::*;


fn sub_f32(crit: &mut Criterion) {
    let mut bench_group = crit.benchmark_group("sub_f32");
    bench_group.sample_size(10);
    
    for Msize in vec![16, 256, 1024, 4096].iter() {
        for Ksize in vec![16, 256, 1024, 4096].iter() {
            //subf32_ndarray
            bench_group.bench_with_input(
                BenchmarkId::new(format!("M{}_K{}_ndarray_f32", Msize, Ksize), 0),
                Msize,
                |bench, msize| {
                    let lhs = Array2::<f32>::ones((*Msize, *Ksize));
                    let rhs = Array2::<f32>::ones((*Msize, *Ksize));
                    bench.iter(|| {
                        black_box(&lhs - &rhs);
                    });
                },
            );
            
            //subf32_owned
            bench_group.bench_with_input(
                BenchmarkId::new(format!("M{}_K{}_owned_f32", Msize, Ksize), 0),
                Msize,
                |bench, msize| {
                    let exec = BlasExecutor::new();
                    bench.iter(|| {
                        let lhs = BlasTensor::ones(vec![*Msize, *Ksize]);
                        let rhs = BlasTensor::ones(vec![*Msize, *Ksize]);
                        black_box(exec.subf32_owned(lhs, rhs));
                    });
                },
            );

            //subf32_side_effect
            bench_group.bench_with_input(
                BenchmarkId::new(format!("M{}_K{}_side_effect_f32", Msize, Ksize), 0),
                Msize,
                |bench, msize| {
                    let lhs = BlasTensor::ones(vec![*Msize, *Ksize]);
                    let rhs = BlasTensor::ones(vec![*Msize, *Ksize]);
                    let mut out = BlasTensor::zeros(vec![*Msize, *Ksize]);
                    let exec = BlasExecutor::new();
                    bench.iter(|| {
                        black_box(exec.subf32_side_effect(&lhs, &rhs, &mut out));
                    });
                },
            );
        }
    }
}

criterion_group!(sub_tests, sub_f32);
criterion_main!(sub_tests);
