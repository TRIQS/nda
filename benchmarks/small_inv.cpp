#include "./bench_common.hpp"

template <int N>
static void inv(benchmark::State &state) {

  nda::matrix<double> W(N, N), Wi(N, N);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) W(i, j) = (i > j ? i + 2.5 * j : i * 0.8 - j);

  while (state.KeepRunning()) {
    benchmark::DoNotOptimize(Wi = inverse(W));
  }
}
BENCHMARK_TEMPLATE(inv, 2);
BENCHMARK_TEMPLATE(inv, 3);
