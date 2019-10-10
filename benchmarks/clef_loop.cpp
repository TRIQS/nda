#include <nda/nda.hpp>
#include <benchmark/benchmark.h>

nda::range_all _;
nda::ellipsis ___;

using namespace nda;

class ABC_1d : public benchmark::Fixture {
  public:
  nda::array<double, 1> a;
  long N;
  void SetUp(const ::benchmark::State &state) {
    N = state.range(0);
    a.resize(N);
    a = 0;
  }

  //void TearDown(const ::benchmark::State &) {}
};

class ABC_3d : public benchmark::Fixture {
  public:
  nda::array<double, 3> a;
  long N;
  void SetUp(const ::benchmark::State &state) {
    N = state.range(0);
    a.resize(N, N, N);
    a = 0;
  }

  //void TearDown(const ::benchmark::State &) {}
};

#define BENCH_ABC_1d(F)                                                                                                                              \
  BENCHMARK_DEFINE_F(ABC_1d, F)(benchmark::State & state) {                                                                                          \
    while (state.KeepRunning()) { F(a, N); }                                                                                                         \
  }                                                                                                                                                  \
  BENCHMARK_REGISTER_F(ABC_1d, F)->Arg(30)->Arg(300);

#define BENCH_ABC_3d(F)                                                                                                                              \
  BENCHMARK_DEFINE_F(ABC_3d, F)(benchmark::State & state) {                                                                                          \
    while (state.KeepRunning()) { F(a, N); }                                                                                                         \
  }                                                                                                                                                  \
  BENCHMARK_REGISTER_F(ABC_3d, F)->Arg(30)->Arg(100);

// -----------------------------------------------------------------------
[[gnu::noinline]] void simple_loop1d(nda::array<double, 1> &a, long N) {
  for (int i = 0; i < N; ++i) a(i) = i + 1;
}

[[gnu::noinline]] void with_clef1d(nda::array<double, 1> &a, long) {
  clef::placeholder<0> i_;
  a(i_) << i_ + 1;
}

// -----------------------------------------------------------------------
[[gnu::noinline]] void simple_loop3d(nda::array<double, 3> &a, long N) {
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      for (int k = 0; k < N; ++k) a(i, j, k) = i - j + N * k;
}

[[gnu::noinline]] void with_clef3d(nda::array<double, 3> &a, long N) {
  clef::placeholder<0> i_;
  clef::placeholder<1> j_;
  clef::placeholder<2> k_;
  a(i_, j_, k_) << i_ - j_ + N * k_;
}

BENCH_ABC_1d(simple_loop1d)
BENCH_ABC_1d(with_clef1d)
BENCH_ABC_3d(simple_loop3d)
BENCH_ABC_3d(with_clef3d)
