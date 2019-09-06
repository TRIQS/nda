#include <nda/nda.hpp>
#include <benchmark/benchmark.h>

nda::range_all _;
nda::ellipsis ___;

using namespace nda;
const int N1 = 100; //, N2 = 1000;

class ABC_2d : public benchmark::Fixture {
  public:
  nda::array<double, 2> a, b, c;

  void SetUp(const ::benchmark::State &state) {
    a.resize(N1, N1);
    b.resize(N1, N1);
    c.resize(N1, N1);
    b = 0;
    c = 0;
  }

  void TearDown(const ::benchmark::State &state) {}
};

#define BENCH_ABC_2d(F)                                                                                                                              \
  BENCHMARK_F(ABC_2d, F)(benchmark::State & state) {                                                                                                 \
    while (state.KeepRunning()) { F(a, b, c); }                                                                                                      \
  }

// -----------------------------------------------------------------------
[[gnu::noinline]] void ex_tmp(nda::array<double, 2> &a, nda::array<double, 2> &b, nda::array<double, 2> &c) { a = 2*b + c; }

[[gnu::noinline]] void ex_tmp_manual_loop(nda::array<double, 2> &a, nda::array<double, 2> &b, nda::array<double, 2> &c) {
  const long l0 = a.shape()[0];
  const long l1 = a.shape()[1];
  for (long i0 = 0; i0 < l0; ++i0)
    for (long i1 = 0; i1 < l1; ++i1) { a(i0, i1) = (2*b + c)(i0, i1); }
}

[[gnu::noinline]] void for_loop(nda::array<double, 2> &a, nda::array<double, 2> &b, nda::array<double, 2> &c) {
  const long l0 = a.shape()[0];
  const long l1 = a.shape()[1];
  for (long i0 = 0; i0 < l0; ++i0)
    for (long i1 = 0; i1 < l1; ++i1) { a(i0, i1) = 2*b(i0, i1) + c(i0, i1); }
}

[[gnu::noinline]] void pointers2dloop(nda::array<double, 2> &a, nda::array<double, 2> &b, nda::array<double, 2> &c) {
  const long s   = a.size();
  const long st0 = a.indexmap().strides()[0];
  const long l0  = a.shape()[0];
  const long l1  = a.shape()[1];

  double *pb = &(b(0, 0));
  double *pa = &(a(0, 0));
  double *pc = &(c(0, 0));
  for (long i0 = 0; i0 < l0; ++i0)
    for (long i1 = 0; i1 < l1; ++i1) { pa[i0 * st0 + i1] = 2*pb[i0 * st0 + i1] + pc[i0 * st0 + i1]; }
}

[[gnu::noinline]] void pointers1dloop(nda::array<double, 2> &a, nda::array<double, 2> &b, nda::array<double, 2> &c) {
  const long s = a.size();

  double *pb = &(b(0, 0));
  double *pa = &(a(0, 0));
  double *pc = &(c(0, 0));
  for (long i = 0; i < s; ++i) { pa[i] = 2*pb[i] + pc[i]; }
}

// -----------------------------------------------------------------------
BENCH_ABC_2d(ex_tmp);
BENCH_ABC_2d(ex_tmp_manual_loop);
BENCH_ABC_2d(for_loop);
BENCH_ABC_2d(pointers2dloop);
BENCH_ABC_2d(pointers1dloop);

BENCHMARK_MAIN();
