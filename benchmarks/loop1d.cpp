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

// --- experiments

[[gnu::noinline]] void pointers2(nda::array<double, 1> &a, nda::array<double, 1> &b, nda::array<double, 1> &c) {
  const long l0         = a.indexmap().lengths()[0];
  double *pb            = &(b(0));
  double *pa            = &(a(0));
  double *__restrict pc = &(c(0));
  for (long i = 0; i < l0; ++i) { pa[i] = pb[i] + pc[i]; }
}

[[gnu::noinline]] void pointers2_withstride(nda::array<double, 1> &a, nda::array<double, 1> &b, nda::array<double, 1> &c) {
  const long l0 = a.indexmap().lengths()[0];
  double *pb    = &(b(0));
  double *pa    = &(a(0));
  double *pc    = &(c(0));
  // for (long i = 0; i < l0; ++i) { pa[i] = pb[i] + pc[i]; }
  for (long i = 0; i < l0; ++i) { pa[a.indexmap().strides()[0] * i] = pb[b.indexmap().strides()[0] * i] + pc[c.indexmap().strides()[0] * i]; }
}

[[gnu::noinline]] void pointers2_withstrideconst(nda::array<double, 1> &a, nda::array<double, 1> &b, nda::array<double, 1> &c) {
  const long l0 = a.indexmap().lengths()[0];
  const long sa = a.indexmap().strides()[0];
  const long sb = b.indexmap().strides()[0];
  const long sc = c.indexmap().strides()[0];
  double *pb    = &(b(0));
  double *pa    = &(a(0));
  double *pc    = &(c(0));
  // for (long i = 0; i < l0; ++i) { pa[i] = pb[i] + pc[i]; }
  for (long i = 0; i < l0; ++i) { pa[sa * i] = pb[sb * i] + pc[sc * i]; }
}

[[gnu::noinline]] void pointers2_withstrideconst_view(nda::array_view<double, 1> a, nda::array_view<double, 1> b, nda::array_view<double, 1> c) {
  const long l0 = a.indexmap().lengths()[0];
  const long sa = a.indexmap().strides()[0];
  const long sb = b.indexmap().strides()[0];
  const long sc = c.indexmap().strides()[0];
  double *pb    = &(b(0));
  double *pa    = &(a(0));
  double *pc    = &(c(0));
  // for (long i = 0; i < l0; ++i) { pa[i] = pb[i] + pc[i]; }
  for (long i = 0; i < l0; ++i) { pa[sa * i] = pb[sb * i] + pc[sc * i]; }
}

struct p_s_t {
  double *p;
  const long s;
};

[[gnu::noinline]] void pointers2_withstrideconst_with_struct(nda::array<double, 1> &a, nda::array<double, 1> &b, nda::array<double, 1> &c) {
  const long l0 = a.indexmap().lengths()[0];
  double *pb    = &(b(0));
  double *pa    = &(a(0));
  double *pc    = &(c(0));
  p_s_t A{pa, a.indexmap().strides()[0]};
  p_s_t B{pb, b.indexmap().strides()[0]};
  p_s_t C{pc, b.indexmap().strides()[0]};
  for (long i = 0; i < l0; ++i) { A.p[A.s * i] = B.p[B.s * i] + C.p[C.s * i]; }
}

[[gnu::noinline]] void pointers2_step2(nda::array<double, 1> &a, nda::array<double, 1> &b, nda::array<double, 1> &c) {
  const long l0         = a.indexmap().lengths()[0];
  double *pb            = &(b(0));
  double *pa            = &(a(0));
  double *__restrict pc = &(c(0));
  for (long i = 0; i < l0; i += 2) { pa[i] = pb[i] + pc[i]; }
}

[[gnu::noinline]] void pointers2_withiterator(nda::array<double, 1> &a, nda::array<double, 1> &b, nda::array<double, 1> &c) {
  const long l0 = a.indexmap().lengths()[0];
  double *pb    = &(b(0));
  double *pa    = &(a(0));
  double *pc    = &(c(0));
  // for (long i = 0; i < l0; ++i) { pa[i] = pb[i] + pc[i]; }
  for (long i = 0; i < l0; ++i) {
    *pa = *pb + *pc;
    pa += a.indexmap().strides()[0];
    pb += b.indexmap().strides()[0];
    pc += c.indexmap().strides()[0];
  }
}

// -----------------------------------------------------------------------
BENCH_ABC_1d(ex_tmp);
BENCH_ABC_1d(ex_tmp_manual_loop);
BENCH_ABC_1d(for_loop);
BENCH_ABC_1d(pointers);
BENCH_ABC_1d(pointers2);
BENCH_ABC_1d(pointers2_withstride);
BENCH_ABC_1d(pointers2_withstrideconst);
BENCH_ABC_1d(pointers2_withstrideconst_view);
BENCH_ABC_1d(pointers2_withstrideconst_with_struct);
//BENCH_ABC_1d(pointers2_step2);
BENCH_ABC_1d(pointers2_withiterator);
