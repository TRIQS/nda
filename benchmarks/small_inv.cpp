#include "./bench_common.hpp"

static void inv_2x2(benchmark::State &state) {

    nda::matrix<double> W(2, 2);
    nda::matrix<double> Wi(2, 2);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) W(i, j) = (i > j ? i + 2.5 * j : i * 0.8 - j);

    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(Wi = inverse(W));
    }
}

BENCHMARK(inv_2x2);

static void inv_3x3(benchmark::State &state) {

    nda::matrix<double> W(3, 3);
    nda::matrix<double> Wi(3, 3);

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) W(i, j) = (i > j ? i + 2.5 * j : i * 0.8 - j);
    
    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(Wi = inverse(W));
    }
}

BENCHMARK(inv_3x3);
