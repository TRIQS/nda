#include "./bench_common.hpp"

class ABC_1d : public benchmark::Fixture {
  public:
  nda::array<double, 1> a;
  long N;
  void SetUp(const ::benchmark::State &state) {
    N = state.range(0);
    a.resize(N);
    //a = 0;
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
    // a = 0;
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
[[gnu::noinline]] void fill_view_gal(nda::basic_array_view<double, 3, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed> v, long ) {
  v = 0;
  benchmark::DoNotOptimize(v(v.extent(0) / 2, 1, 2));
}

[[gnu::noinline]] void fill_view_contiguous(nda::basic_array_view<double, 3, nda::C_layout, 'A', nda::default_accessor, nda::borrowed> v, long) {
  v = 0;
  benchmark::DoNotOptimize(v(v.extent(0) / 2, 1, 2));
}

[[gnu::noinline]] void fill_view_contiguous_restrict(nda::basic_array_view<double, 3, nda::C_layout, 'A', nda::no_alias_accessor, nda::borrowed> v, long) {
  v = 0;
  benchmark::DoNotOptimize(v(v.extent(0) / 2, 1, 2));
}

// -----------------------------------------------------------------------

BENCH_ABC_3d(fill_view_gal);
BENCH_ABC_3d(fill_view_contiguous);
BENCH_ABC_3d(fill_view_contiguous_restrict);
