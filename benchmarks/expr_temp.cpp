#include <nda.hpp>
#include <benchmark/benchmark.h>

nda::range_all _;
nda::ellipsis ___;

using namespace nda;
const int N1 = 1000, N2 = 1000;

// -------------------------------- 1d ---------------------------------------

static void add(benchmark::State &state) {
  nda::array<double, 1> a(N1);
  nda::array<double, 1> b(N1);
  nda::array<double, 1> c(N1);
  b = 0;
  c = 0;

  while (state.KeepRunning()) { benchmark::DoNotOptimize(a = 2*b + c); }
}
BENCHMARK(add);

static void manual_add(benchmark::State &state) {
  nda::array<double, 1> a(N1);
  nda::array<double, 1> b(N1);
  nda::array<double, 1> c(N1);
  b             = 0;
  c             = 0;
  const long l0 = a.indexmap().lengths()[0];
  double *pa    = &(a(0));
  double *pb    = &(b(0));
  double *pc    = &(c(0));

  while (state.KeepRunning()) {
    for (long i = 0; i < l0; ++i) benchmark::DoNotOptimize(pa[i] = 2*pb[i] + pc[i]);
  }
}
BENCHMARK(manual_add);
BENCHMARK_MAIN();
