
#include <nda.hpp>
#include <benchmark/benchmark.h>

nda::range_all _;
nda::ellipsis ___;

using namespace nda;
const int N1 = 10; //, N2 = 1000;

class ABC_3d : public benchmark::Fixture {
  public:
  nda::array<double, 3> a, b, c;

  void SetUp(const ::benchmark::State &state) {
    a.resize(N1, N1, N1);
    b.resize(N1, N1, N1);
    c.resize(N1, N1, N1);
    b = 0;
    c = 0;
  }

  void TearDown(const ::benchmark::State &state) {}
};

#define BENCH_ABC_3d(F)                                                                                                                              \
  BENCHMARK_F(ABC_3d, F)(benchmark::State & state) {                                                                                                 \
    while (state.KeepRunning()) { F(a, b, c); }                                                                                                      \
  }

// retest with view
// -----------------------------------------------------------------------
[[gnu::noinline]] void ex_tmp(nda::array<double, 3> &a, nda::array<double, 3> &b, nda::array<double, 3> &c) {
 
  static_assert(guarantee::has_contiguous(get_guarantee<std::decay_t<decltype(2*b+c)>>), "EEE");
  a = 2*b + c; }

[[gnu::noinline]] void ex_tmp_manual_loop(nda::array<double, 3> &a, nda::array<double, 3> &b, nda::array<double, 3> &c) {
  const long l0 = a.shape()[0];
  const long l1 = a.shape()[0];
  const long l2  = a.shape()[2];
  for (long i0 = 0; i0 < l0; ++i0)
    for (long i1 = 0; i1 < l1; ++i1)
      for (long i2 = 0; i2 < l2; ++i2) { a(i0, i1, i2) = (2 * b + c)(i0, i1, i2); }
}

[[gnu::noinline]] void for_loop(nda::array<double, 3> &a, nda::array<double, 3> &b, nda::array<double, 3> &c) {
  const long l0 = a.shape()[0];
  const long l1 = a.shape()[0];
  const long l2  = a.shape()[2];
  for (long i0 = 0; i0 < l0; ++i0)
    for (long i1 = 0; i1 < l1; ++i1)
      for (long i2 = 0; i2 < l2; ++i2) { a(i0, i1, i2) = 2 * b(i0, i1, i2) + c(i0, i1, i2); }
}

[[gnu::noinline]] void pointers3dloop(nda::array<double, 3> &a, nda::array<double, 3> &b, nda::array<double, 3> &c) {
  const long st0 = a.indexmap().strides()[0];
  const long st1 = a.indexmap().strides()[1];
  const long st2 = a.indexmap().strides()[2];
  const long l0  = a.shape()[0];
  const long l1  = a.shape()[1];
  const long l2  = a.shape()[2];

  double *pb = &(b(0, 0, 0));
  double *pa = &(a(0, 0, 0));
  double *pc = &(c(0, 0, 0));
  for (long i0 = 0; i0 < l0; ++i0)
    for (long i1 = 0; i1 < l1; ++i1)
      //for (long i2 = 0; i2 < l2; ++i2) { pa[i0 * st0 + i1 * st1 + i2*st2] = 2 * pb[i0 * st0 + i1 * st1 + i2*st2] + pc[i0 * st0 + i1 * st1 + i2*st2]; }
      for (long i2 = 0; i2 < l2; ++i2) { pa[i0 * st0 + i1 * st1 + i2] = 2 * pb[i0 * st0 + i1 * st1 + i2] + pc[i0 * st0 + i1 * st1 + i2]; }
}

[[gnu::noinline]] void pointers1dloop(nda::array<double, 3> &a, nda::array<double, 3> &b, nda::array<double, 3> &c) {
  const long s = a.size();

  double *pb = &(b(0, 0, 0));
  double *pa = &(a(0, 0, 0));
  double *pc = &(c(0, 0, 0));
  //for (long i = 0; i < s; ++i) { a.storage()[i] =  b.storage()[i] + c.storage()[i]; }
  for (long i = 0; i < s; ++i) { pa[i] =  2*pb[i] + pc[i]; }
}

// -----------------------------------------------------------------------
BENCH_ABC_3d(ex_tmp);
BENCH_ABC_3d(ex_tmp_manual_loop);
BENCH_ABC_3d(for_loop);
BENCH_ABC_3d(pointers3dloop);
BENCH_ABC_3d(pointers1dloop);

BENCHMARK_MAIN();
