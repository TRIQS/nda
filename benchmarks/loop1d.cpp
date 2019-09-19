#include <nda/nda.hpp>
#include <benchmark/benchmark.h>

nda::range_all _;
nda::ellipsis ___;

using namespace nda;
const int N1 = 10000; //, N2 = 1000;

class ABC_1d : public benchmark::Fixture {
  public:
  nda::array<double, 1> a, b, c;

  void SetUp(const ::benchmark::State &) {
    a.resize(N1);
    b.resize(N1);
    c.resize(N1);
    b = 0;
    c = 0;
  }

  void TearDown(const ::benchmark::State &) {}
};

#define BENCH_ABC_1d(F)                                                                                                                              \
  BENCHMARK_F(ABC_1d, F)(benchmark::State & state) {                                                                                                 \
    while (state.KeepRunning()) { F(a, b, c); }                                                                                                      \
  }

// -----------------------------------------------------------------------
[[gnu::noinline]] void ex_tmp(nda::array<double, 1> &a, nda::array<double, 1> &b, nda::array<double, 1> &c) { a() = b + c; }

[[gnu::noinline]] void ex_tmp_manual_loop(nda::array<double, 1> &a, nda::array<double, 1> &b, nda::array<double, 1> &c) {
  const long l0 = a.indexmap().lengths()[0];
  for (long i = 0; i < l0; ++i) { a(i) = (b + c)(i); }
}

[[gnu::noinline]] void for_loop(nda::array<double, 1> &a, nda::array<double, 1> &b, nda::array<double, 1> &c) {
  const long l0 = a.indexmap().lengths()[0];
  for (long i = 0; i < l0; ++i) { a(i) = b(i) + c(i); } // () is bad  x2!
}

[[gnu::noinline]] void pointers(nda::array<double, 1> &a, nda::array<double, 1> &b, nda::array<double, 1> &c) {
  const long l0 = a.indexmap().lengths()[0];
  double *pb    = &(b(0));
  double *pa    = &(a(0));
  double *pc    = &(c(0));
  for (long i = 0; i < l0; ++i) { pa[i] = pb[i] + pc[i]; }
}

// -----------------------------------------------------------------------
BENCH_ABC_1d(ex_tmp);
BENCH_ABC_1d(ex_tmp_manual_loop);
BENCH_ABC_1d(for_loop);
BENCH_ABC_1d(pointers);
