#include "./bench_common.hpp"

template <int N, long dim>
static void inv(benchmark::State &state) {
  nda::array<double, 3> W(dim, N, N), Wi(dim, N, N);
  for (int k = 0; k < dim; ++k) {
      for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) W(k, i, j) = (i > j ? 0.5 + i + 2.5 * j : i * 0.8 - j - 0.5);
  }

  while (state.KeepRunning()) {
    Wi = inverse(W);
  }
}

BENCHMARK_TEMPLATE(inv, 1, 100);
BENCHMARK_TEMPLATE(inv, 1, 10000);
BENCHMARK_TEMPLATE(inv, 1, 1000000);
BENCHMARK_TEMPLATE(inv, 2, 100);
BENCHMARK_TEMPLATE(inv, 2, 10000);
BENCHMARK_TEMPLATE(inv, 2, 1000000);
BENCHMARK_TEMPLATE(inv, 3, 100);
BENCHMARK_TEMPLATE(inv, 3, 10000);
BENCHMARK_TEMPLATE(inv, 3, 1000000);
