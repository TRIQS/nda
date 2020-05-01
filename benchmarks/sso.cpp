#include <nda/nda.hpp>
#include <benchmark/benchmark.h>

static void dyn_alloc(benchmark::State &state) {
  const int N = state.range(0);

  while (state.KeepRunning()) {
  nda::array<double, 1> A(N);
    for (int i = 0; i < N - 1; ++i) benchmark::DoNotOptimize(A(i) = i);//fnt(i));
    for (int i = 0; i < N - 1; ++i) benchmark::DoNotOptimize(A(i) = i);//fnt(i));
  }
}
BENCHMARK(dyn_alloc)->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(10)->Arg(15);//->Arg(30)->Arg(50)->Arg(100)->Arg(500);

static void dyn_alloc_loop_only(benchmark::State &state) {
  const int N = state.range(0);
  nda::array<double, 1> A(N);

  while (state.KeepRunning()) {
    for (int i = 0; i < N - 1; ++i) benchmark::DoNotOptimize(A(i) = i);//fnt(i));
    for (int i = 0; i < N - 1; ++i) benchmark::DoNotOptimize(A(i) = i);//fnt(i));
  }
}
BENCHMARK(dyn_alloc_loop_only)->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(10)->Arg(15);//->Arg(30)->Arg(50)->Arg(100)->Arg(500);


static void stack_alloc(benchmark::State &state) {
  const int N = state.range(0);

  while (state.KeepRunning()) {
  nda::stack_array<double, 1, nda::static_extents(15)> A(N);
    for (int i = 0; i < N - 1; ++i) benchmark::DoNotOptimize(A(i) = i);//fnt(i));
    for (int i = 0; i < N - 1; ++i) benchmark::DoNotOptimize(A(i) = i);//fnt(i));
  }
}
BENCHMARK(stack_alloc)->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(10)->Arg(15);//->Arg(30)->Arg(50)->Arg(100);//->Arg(500);

static void stack_alloc_loop_only(benchmark::State &state) {
  const int N = state.range(0);
  nda::stack_array<double, 1, nda::static_extents(15)> A(N);

  while (state.KeepRunning()) {
    for (int i = 0; i < N - 1; ++i) benchmark::DoNotOptimize(A(i) = i);//fnt(i));
    for (int i = 0; i < N - 1; ++i) benchmark::DoNotOptimize(A(i) = i);//fnt(i));
  }
}
BENCHMARK(stack_alloc_loop_only)->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(10)->Arg(15);//->Arg(30)->Arg(50)->Arg(100);//->Arg(500);


// --------  2d------------

static void dyn_alloc2d(benchmark::State &state) {
  const int N = state.range(0);

  while (state.KeepRunning()) {
  nda::array<double, 2> A(N,N);
    for (int i = 0; i < N - 1; ++i) 
    for (int j = 0; j < N - 1; ++j) benchmark::DoNotOptimize(A(i,j)  = i+2*j);//fnt(i));
  }
}
BENCHMARK(dyn_alloc2d)->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(10)->Arg(15);//->Arg(30)->Arg(50)->Arg(100)->Arg(500);

static void stack_alloc2d(benchmark::State &state) {
  const int N = state.range(0);

  while (state.KeepRunning()) {
  nda::stack_array<double, 2, nda::static_extents(15,15)> A;
    for (int i = 0; i < N - 1; ++i) 
    for (int j = 0; j < N - 1; ++j) benchmark::DoNotOptimize(A(i,j)  = i+2*j);//fnt(i));
  }
}
BENCHMARK(stack_alloc2d)->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(10)->Arg(15);//->Arg(30)->Arg(50)->Arg(100);//->Arg(500);

static void dyn_alloc2d_loop_only(benchmark::State &state) {
  const int N = state.range(0);
  nda::array<double, 2> A(N,N);

  while (state.KeepRunning()) {
    for (int i = 0; i < N - 1; ++i) 
    for (int j = 0; j < N - 1; ++j) benchmark::DoNotOptimize(A(i,j)  = i+2*j);//fnt(i));
  }
}
BENCHMARK(dyn_alloc2d_loop_only)->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(10)->Arg(15);//->Arg(30)->Arg(50)->Arg(100)->Arg(500);

static void stack_alloc2d_loop_only(benchmark::State &state) {
  const int N = state.range(0);
  nda::stack_array<double, 2, nda::static_extents(15,15)> A;

  while (state.KeepRunning()) {
    for (int i = 0; i < N - 1; ++i) 
    for (int j = 0; j < N - 1; ++j) benchmark::DoNotOptimize(A(i,j)  = i+2*j);//fnt(i));
  }
}
BENCHMARK(stack_alloc2d_loop_only)->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(10)->Arg(15);//->Arg(30)->Arg(50)->Arg(100);//->Arg(500);


