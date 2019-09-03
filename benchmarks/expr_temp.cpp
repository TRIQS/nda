#include <nda.hpp>
#include <benchmark/benchmark.h>

nda::range_all _;
nda::ellipsis ___;

using namespace nda;
const int N1 = 1000, N2 = 1000;

// -------------------------------- 1d ---------------------------------------

// Does not work. Optimized ...
//static void add(benchmark::State &state) {
  //nda::array<double, 1> a(N1);
  //nda::array<double, 1> b(N1);
  //nda::array<double, 1> c(N1);
  //b = 0;
  //c = 0;

  //while (state.KeepRunning()) {
    //benchmark::DoNotOptimize(a);
    //a = 2 * b + c;
  //}
//}
//BENCHMARK(add);

//----------------------------------------
static void add2(benchmark::State &state) {
  nda::array<double, 1> a(N1);
  nda::array<double, 1> b(N1);
  const long l0 = a.shape()[0];
  nda::array<double, 1> c(N1);
  b = 0;
  c = 0;

  auto l = [&](auto const &... args) { benchmark::DoNotOptimize(a(args...) = (2 * b + c)(args...)); };

  while (state.KeepRunning()) {
    for (long i = 0; i < l0; ++i) {
      // benchmark::DoNotOptimize(a(i) = (2*b +c)(i)); //l(i); // nda::for_each(a.shape(), l);
      benchmark::DoNotOptimize(a);
      l(i); // nda::for_each(a.shape(), l);
      //benchmark::DoNotOptimize(a = 2 * b + c);
    }
  }
}
BENCHMARK(add2);

//----------------------------------------

static void add_for1(benchmark::State &state) {
  nda::array<double, 1> a(N1);
  nda::array<double, 1> b(N1);
  nda::array<double, 1> c(N1);
  const long l0 = a.shape()[0];
  b             = 0;
  c             = 0;

  while (state.KeepRunning()) {
    for (long i = 0; i < l0; ++i) {
      benchmark::DoNotOptimize(a);
      a(i) = (2 * b + c)(i);
    }
  }
}
BENCHMARK(add_for1);

//----------------------------------------

static void add_for(benchmark::State &state) {
  nda::array<double, 1> a(N1);
  nda::array<double, 1> b(N1);
  nda::array<double, 1> c(N1);
  const long l0 = a.shape()[0];
  b             = 0;
  c             = 0;

  while (state.KeepRunning()) {
    for (long i = 0; i < l0; ++i) {
      benchmark::DoNotOptimize(a);
      //a.data_start()[i] = c.data_start()[i];
      //a.data_start()[i] = 2 * b.data_start()[i] + c.data_start()[i];
      a(i) = b(i) + c(i);
      //a(i) = c(i);
    }
  }
}
BENCHMARK(add_for);

//---------------------------

static void manual_add(benchmark::State &state) {
  nda::array<double, 1> a(N1);
  nda::array<double, 1> b(N1);
  nda::array<double, 1> c(N1);
  b             = 0;
  c             = 0;
  const long l0 = a.shape()[0];
  double *pa    = &(a(0));
  double *pb    = &(b(0));
  double *pc    = &(c(0));

  while (state.KeepRunning()) {
    for (long i = 0; i < l0; ++i) {
      //benchmark::DoNotOptimize(pa[i] =  pc[i]);
      benchmark::DoNotOptimize(pa[i] = pb[i] + pc[i]);
    }
  }
}
BENCHMARK(manual_add);
BENCHMARK_MAIN();
