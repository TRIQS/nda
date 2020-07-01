#include "./bench_common.hpp"

// using VALUE_TYPE=double ;
using VALUE_TYPE = int;
inline VALUE_TYPE fnt(size_t i) { return i * (i + 2.0) * (i - 8.0); }

static void view1_reference(benchmark::State &state) {
  const int N = state.range(0);
  nda::array<double, 3> A(N, 2, 2);

  while (state.KeepRunning()) {
    for (int i = 0; i < N - 1; ++i) benchmark::DoNotOptimize(A(i, 0, 0) = fnt(i));
  }
}
BENCHMARK(view1_reference)->Arg(3)->Arg(5)->Arg(10)->Arg(30)->Arg(100)->Arg(300);

// ---------------------------------

static void view1(benchmark::State &state) {
  const int N = state.range(0);
  nda::array<double, 3> A(N, 2, 2);

  while (state.KeepRunning()) {
    for (int i = 0; i < N - 1; ++i) benchmark::DoNotOptimize(A(i, _, _)(0, 0) = fnt(i));
  }
}
BENCHMARK(view1)->Arg(3)->Arg(5)->Arg(10)->Arg(30)->Arg(100)->Arg(300);

// ==========================================================

static void view2_reference(benchmark::State &state) {
  const int N = state.range(0);
  nda::array<double, 4> A(N, N, 2, 2);

  while (state.KeepRunning()) {
    for (int i = 0; i < N - 1; ++i) benchmark::DoNotOptimize(A(i, i, 0, 0) = fnt(i));
  }
}
BENCHMARK(view2_reference)->Arg(3)->Arg(5)->Arg(10)->Arg(30)->Arg(100)->Arg(300);

// ---------------------------------

static void view2(benchmark::State &state) {
  const int N = state.range(0);
  nda::array<double, 4> A(N, N, 2, 2);

  while (state.KeepRunning()) {
    for (int i = 0; i < N - 1; ++i) benchmark::DoNotOptimize(A(i, i, _, _)(0, 0) = fnt(i));
  }
}
BENCHMARK(view2)->Arg(3)->Arg(5)->Arg(10)->Arg(30)->Arg(100)->Arg(300);

// ---------------------------------

static void view2_ellipsis(benchmark::State &state) {
  const int N = state.range(0);
  nda::array<double, 4> A(N, N, 2, 2);

  while (state.KeepRunning()) {
    for (int i = 0; i < N - 1; ++i) benchmark::DoNotOptimize(A(i, i, ___)(0, 0) = fnt(i));
  }
}
BENCHMARK(view2_ellipsis)->Arg(3)->Arg(5)->Arg(10)->Arg(30)->Arg(100)->Arg(300);

// ==========================================================

static void view2b_reference(benchmark::State &state) {
  const int N = state.range(0);
  nda::array<double, 4> A(3, 3, N, N);

  while (state.KeepRunning()) {
    for (int i = 0; i < N - 1; ++i) benchmark::DoNotOptimize(A(1, 2, i, i) = fnt(i));
  }
}
BENCHMARK(view2b_reference)->Arg(3)->Arg(5)->Arg(10)->Arg(30)->Arg(100)->Arg(300);

// ---------------------------------

static void view2b(benchmark::State &state) {
  const int N = state.range(0);
  nda::array<double, 4> A(3, 3, N, N);

  while (state.KeepRunning()) {
    for (int i = 0; i < N - 1; ++i) benchmark::DoNotOptimize(A(_, _, i, i)(1, 2) = fnt(i));
  }
}
BENCHMARK(view2b)->Arg(3)->Arg(5)->Arg(10)->Arg(30)->Arg(100)->Arg(300);

// ==========================================================

static void view2M_reference(benchmark::State &state) {
  const int N = state.range(0);
  const int P = 10;
  nda::array<double, 4> A(N, P, N, P);

  while (state.KeepRunning()) {
    for (int i = 0; i < N - 1; ++i)
      for (int j = 0; j < P; ++j)
        for (int k = 0; k < P; ++k) benchmark::DoNotOptimize(A(i, j, i, k) = fnt(i));
  }
}
BENCHMARK(view2M_reference)->Arg(3)->Arg(5)->Arg(10)->Arg(30)->Arg(100)->Arg(300);

// ---------------------------------

static void view2M(benchmark::State &state) {
  const int N = state.range(0);
  const int P = 10;
  nda::array<double, 4> A(N, P, N, P);

  while (state.KeepRunning()) {
    for (int i = 0; i < N - 1; ++i)
      for (int j = 0; j < P; ++j)
        for (int k = 0; k < P; ++k) benchmark::DoNotOptimize(A(i, _, i, _)(j, k) = fnt(i));
  }
}
BENCHMARK(view2M)->Arg(3)->Arg(5)->Arg(10)->Arg(30)->Arg(100)->Arg(300);

// ==========================================================

static void view3_reference(benchmark::State &state) {
  const int N = state.range(0);
  nda::array<double, 5> A(N, N, N, 2, 2);

  while (state.KeepRunning()) {
    for (int i = 0; i < N - 1; ++i) benchmark::DoNotOptimize(A(i, i, i, 0, 0) = fnt(i));
  }
}
BENCHMARK(view3_reference)->Arg(3)->Arg(5)->Arg(10)->Arg(30)->Arg(100)->Arg(300);

// ---------------------------------

static void view3(benchmark::State &state) {
  const int N = state.range(0);
  nda::array<double, 5> A(N, N, N, 2, 2);

  while (state.KeepRunning()) {
    for (int i = 0; i < N - 1; ++i) benchmark::DoNotOptimize(A(i, i, i, _, _)(0, 0) = fnt(i));
  }
}
BENCHMARK(view3)->Arg(3)->Arg(5)->Arg(10)->Arg(30)->Arg(100)->Arg(300);

// ==========================================================
static void view8_reference(benchmark::State &state) {
  const int N = state.range(0);
  nda::array<double, 8> A(N, N, N, N, 2, 2, 2, 2);

  while (state.KeepRunning()) {
    for (int i = 0; i < N - 1; ++i) benchmark::DoNotOptimize(A(i, i, i, i, 0, 0, 0, 0) = fnt(i));
  }
}
BENCHMARK(view8_reference)->Arg(3)->Arg(5)->Arg(10);

// ---------------------------------

static void view8(benchmark::State &state) {
  const int N = state.range(0);
  nda::array<double, 8> A(N, N, N, N, 2, 2, 2, 2);

  while (state.KeepRunning()) {
    for (int i = 0; i < N - 1; ++i) benchmark::DoNotOptimize(A(i, i, i, i, ___)(0, 0, 0, 0) = fnt(i));
  }
}
BENCHMARK(view8)->Arg(3)->Arg(5)->Arg(10);
