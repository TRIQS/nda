#include "./bench_common.hpp"

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
  BENCHMARK_REGISTER_F(ABC_1d, F)->Arg(3)->Arg(5)->Arg(10)->Arg(30)->Arg(100)->Arg(300)

#define BENCH_ABC_3d(F)                                                                                                                              \
  BENCHMARK_DEFINE_F(ABC_3d, F)(benchmark::State & state) {                                                                                          \
    while (state.KeepRunning()) { F(a, N); }                                                                                                         \
  }                                                                                                                                                  \
  BENCHMARK_REGISTER_F(ABC_3d, F)->Arg(3)->Arg(5)->Arg(10)->Arg(30)->Arg(100)->Arg(300)

// -----------------------------------------------------------------------
[[gnu::noinline]] void clef1d_reference(nda::array_contiguous_view<double, 1> a, const long N) {
  for (int i = 0; i < N; ++i) benchmark::DoNotOptimize(a(i) = i + 1);
}

[[gnu::noinline]] void clef1d(nda::array<double, 1> &a, long) {
  clef::placeholder<0> i_;
  a(i_) << i_ + 1;
}

[[gnu::noinline]] void clef1d_foreach_rewrite(nda::array<double, 1> &a, long) {
  nda::for_each(a.shape(), [&a](long i) { benchmark::DoNotOptimize(a(i) = i + 1); });
}

// -----------------------------------------------------------------------
[[gnu::noinline]] void clef3d_reference(nda::array<double, 3> &a, long N) {
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      for (int k = 0; k < N; ++k) a(i, j, k) = i - j + N * k;
}

[[gnu::noinline]] void clef3d(nda::array<double, 3> &a, long N) {
  clef::placeholder<0> i_;
  clef::placeholder<1> j_;
  clef::placeholder<2> k_;
  a(i_, j_, k_) << i_ - j_ + N * k_;
}

BENCH_ABC_1d(clef1d_reference);
BENCH_ABC_1d(clef1d);
BENCH_ABC_1d(clef1d_foreach_rewrite);
BENCH_ABC_3d(clef3d_reference);
BENCH_ABC_3d(clef3d);
