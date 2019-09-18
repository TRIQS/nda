#include <nda/array.hpp>
#include <benchmark/benchmark.h>

nda::range_all _;
nda::ellipsis ___;

// using VALUE_TYPE=double ;
using VALUE_TYPE = int;
inline VALUE_TYPE fnt(size_t i) { return i * (i + 2.0) * (i - 8.0); }

static void view1(benchmark::State &state) {
  const int N = state.range(0);
  nda::array<double, 3> A(N, 2, 2);

  while (state.KeepRunning()) {
    for (int i = 0; i < N - 1; ++i) benchmark::DoNotOptimize(A(i, _, _)(0, 0) = fnt(i));
  }
}
BENCHMARK(view1)->Arg(30)->Arg(300);

// ---------------------------------

static void direct1(benchmark::State &state) {
  const int N = state.range(0);
  nda::array<double, 3> A(N, 2, 2);

  while (state.KeepRunning()) {
    for (int i = 0; i < N - 1; ++i) benchmark::DoNotOptimize(A(i, 0, 0) = fnt(i));
  }
}
BENCHMARK(direct1)->Arg(30)->Arg(300);

// ==========================================================

static void view2(benchmark::State &state) {
  const int N = state.range(0);
  nda::array<double, 4> A(N, N, 2, 2);

  while (state.KeepRunning()) {
    for (int i = 0; i < N - 1; ++i) benchmark::DoNotOptimize(A(i, i, _, _)(0, 0) = fnt(i));
  }
}
BENCHMARK(view2)->Arg(30)->Arg(300);

// ---------------------------------

static void view2_ellipsis(benchmark::State &state) {
  const int N = state.range(0);
  nda::array<double, 4> A(N, N, 2, 2);

  while (state.KeepRunning()) {
    for (int i = 0; i < N - 1; ++i) benchmark::DoNotOptimize(A(i, i, ___)(0, 0) = fnt(i));
  }
}
BENCHMARK(view2_ellipsis)->Arg(30)->Arg(300);

// ---------------------------------

static void direct2(benchmark::State &state) {
  const int N = state.range(0);
  nda::array<double, 4> A(N, N, 2, 2);

  while (state.KeepRunning()) {
    for (int i = 0; i < N - 1; ++i) benchmark::DoNotOptimize(A(i, i, 0, 0) = fnt(i));
  }
}
BENCHMARK(direct2)->Arg(30)->Arg(300);

// ==========================================================

static void view2A(benchmark::State &state) {
  const int N = state.range(0);
  nda::array<double, 4> A(3, 3, N, N);

  while (state.KeepRunning()) {
    for (int i = 0; i < N - 1; ++i) benchmark::DoNotOptimize(A(_, _, i, i)(1, 2) = fnt(i));
  }
}
BENCHMARK(view2A)->Arg(30)->Arg(300);

// ---------------------------------

static void direct2A(benchmark::State &state) {
  const int N = state.range(0);
  nda::array<double, 4> A(3, 3, N, N);

  while (state.KeepRunning()) {
    for (int i = 0; i < N - 1; ++i) benchmark::DoNotOptimize(A(1, 2, i, i) = fnt(i));
  }
}
BENCHMARK(direct2A)->Arg(30)->Arg(300);

// ==========================================================

static void view2M(benchmark::State &state) {
  const int N = state.range(0);
  const int P = 10;
  nda::array<double, 4> A(N, N, P, P);

  while (state.KeepRunning()) {
    for (int i = 0; i < N - 1; ++i)
      for (int j = 0; j < P; ++j)
        for (int k = 0; k < P; ++k) benchmark::DoNotOptimize(A(i, _, i, _)(j, k) = fnt(i));
  }
}
BENCHMARK(view2M)->Arg(30)->Arg(300);

// ---------------------------------

static void direct2M(benchmark::State &state) {
  const int N = state.range(0);
  const int P = 10;
  nda::array<double, 4> A(N, P, N, P);

  while (state.KeepRunning()) {
    for (int i = 0; i < N - 1; ++i)
      for (int j = 0; j < P; ++j)
        for (int k = 0; k < P; ++k) benchmark::DoNotOptimize(A(i, j, i, k) = fnt(i));
  }
}
BENCHMARK(direct2M)->Arg(30)->Arg(300);

// ==========================================================

static void view3(benchmark::State &state) {
  const int N = state.range(0);
  nda::array<double, 5> A(N, N, N, 2, 2);

  while (state.KeepRunning()) {
    for (int i = 0; i < N - 1; ++i) benchmark::DoNotOptimize(A(i, i, i, _, _)(0, 0) = fnt(i));
  }
}
BENCHMARK(view3)->Arg(30)->Arg(300);

// ---------------------------------

static void direct3(benchmark::State &state) {
  const int N = state.range(0);
  nda::array<double, 5> A(N, N, N, 2, 2);

  while (state.KeepRunning()) {
    for (int i = 0; i < N - 1; ++i) benchmark::DoNotOptimize(A(i, i, i, 0, 0) = fnt(i));
  }
}
BENCHMARK(direct3)->Arg(30)->Arg(300);

// ==========================================================

static void view8(benchmark::State &state) {
  const int N = state.range(0);
  nda::array<double, 8> A(N, N, N, N, 2, 2, 2, 2);

  while (state.KeepRunning()) {
    for (int i = 0; i < N - 1; ++i) benchmark::DoNotOptimize(A(i, i, i, i, ___)(0, 0, 0, 0) = fnt(i));
  }
}
BENCHMARK(view8)->Arg(30);

// ---------------------------------

static void direct8(benchmark::State &state) {
  const int N = state.range(0);
  nda::array<double, 8> A(N, N, N, N, 2, 2, 2, 2);

  while (state.KeepRunning()) {
    for (int i = 0; i < N - 1; ++i) benchmark::DoNotOptimize(A(i, i, i, i, 0, 0, 0, 0) = fnt(i));
  }
}
BENCHMARK(direct8)->Arg(30);

// ==========================================================

//static void direct_1d(benchmark::State &state) {
  //const int N = state.range(0);
  //nda::array<double, 1> A(N);
  ////A() = 0;

  //while (state.KeepRunning()) {
    //for (int i = 0; i < N - 1; ++i) A(i) = fnt(i);
  //}
//}
//BENCHMARK(direct_1d)->Arg(30)->Arg(300);
