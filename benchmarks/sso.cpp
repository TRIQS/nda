// Copyright (c) 2020 Simons Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "./bench_common.hpp"

static void dyn_alloc(benchmark::State &state) {
  const int N = state.range(0);

  while (state.KeepRunning()) { nda::array<double, 1> A(N); }
}
BENCHMARK(dyn_alloc)->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(10)->Arg(15); //->Arg(30)->Arg(50)->Arg(100)->Arg(500);

//static void dyn_alloc_and_loop(benchmark::State &state) {
//const int N = state.range(0);

//while (state.KeepRunning()) {
//nda::array<double, 1> A(N);
//for (int i = 0; i < N - 1; ++i) benchmark::DoNotOptimize(A(i) = i);//fnt(i));
//}
//}
//BENCHMARK(dyn_alloc_and_loop)->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(10)->Arg(15);//->Arg(30)->Arg(50)->Arg(100)->Arg(500);

static void dyn_loop_only(benchmark::State &state) {
  const int N = state.range(0);
  nda::array<double, 1> A(N);

  while (state.KeepRunning()) {
    for (int i = 0; i < N - 1; ++i) benchmark::DoNotOptimize(A(i) = i); //fnt(i));
  }
}
BENCHMARK(dyn_loop_only)->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(10)->Arg(15); //->Arg(30)->Arg(50)->Arg(100)->Arg(500);

// ------------------

static void sso_alloc(benchmark::State &state) {
  const int N = state.range(0);
  using a_t   = nda::basic_array<long, 1, nda::C_layout, 'A', nda::sso<100>>;

  while (state.KeepRunning()) {
    a_t A(N);
    benchmark::DoNotOptimize(A(0));
  }
}
BENCHMARK(sso_alloc)->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(10)->Arg(15); //->Arg(30)->Arg(50)->Arg(100);//->Arg(500);

/*static void sso_alloc_handle(benchmark::State &state) {*/
//const int N = state.range(0);
//using a_t = nda::basic_array<long, 1, nda::C_layout, 'A', nda::sso<100>>;

//while (state.KeepRunning()) {
//benchmark::DoNotOptimize(A(0));

//}
//}
//BENCHMARK(sso_alloc_handle)->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(10)->Arg(15);//->Arg(30)->Arg(50)->Arg(100);//->Arg(500);

//static void sso_alloc_and_loop(benchmark::State &state) {
//const int N = state.range(0);
//using a_t = nda::basic_array<long, 1, nda::C_layout, 'A', nda::sso<15>>;

//while (state.KeepRunning()) {
//a_t A(N);
//for (int i = 0; i < N - 1; ++i) benchmark::DoNotOptimize(A(i) = i);//fnt(i));
//}
//}
//BENCHMARK(sso_alloc_and_loop)->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(10)->Arg(15);//->Arg(30)->Arg(50)->Arg(100);//->Arg(500);

static void sso_loop_only(benchmark::State &state) {
  const int N = state.range(0);
  using a_t   = nda::basic_array<long, 1, nda::C_layout, 'A', nda::sso<100>>;
  a_t A(N);

  while (state.KeepRunning()) {
    for (int i = 0; i < N - 1; ++i) benchmark::DoNotOptimize(A(i) = i); //fnt(i));
  }
}
BENCHMARK(sso_loop_only)->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(10)->Arg(15)->Arg(100); //->Arg(30)->Arg(50)->Arg(100);//->Arg(500);

static void sso_loop_only_restrict(benchmark::State &state) {
  const int N = state.range(0);
  using a_t   = nda::basic_array<long, 1, nda::C_layout, 'A', nda::sso<100>>;
  a_t A(N);
  nda::basic_array_view<long, 1, nda::C_layout, 'A', nda::no_alias_accessor, nda::borrowed> v{A};
  //nda::basic_array_view<long, 1, nda::C_layout, 'A', nda::default_accessor, nda::borrowed> v{A};

  while (state.KeepRunning()) {
    for (int i = 0; i < N - 1; ++i) benchmark::DoNotOptimize(v(i) = i); //fnt(i));
  }
}
BENCHMARK(sso_loop_only_restrict)->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(10)->Arg(15)->Arg(100); //->Arg(30)->Arg(50)->Arg(100);//->Arg(500);

// ------------------

static void stack_alloc(benchmark::State &state) {
  const int N = state.range(0);

  while (state.KeepRunning()) {
    nda::stack_array<long, 1, nda::static_extents(15)> A(N);
    benchmark::DoNotOptimize(A(0));
  }
}
BENCHMARK(stack_alloc)->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(10)->Arg(15); //->Arg(30)->Arg(50)->Arg(100);//->Arg(500);

//static void stack_alloc_and_loop(benchmark::State &state) {
//const int N = state.range(0);

//while (state.KeepRunning()) {
//nda::stack_array<long, 1, nda::static_extents(15)> A(N);
//for (int i = 0; i < N - 1; ++i) benchmark::DoNotOptimize(A(i) = i);//fnt(i));
//}
//}
//BENCHMARK(stack_alloc_and_loop)->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(10)->Arg(15);//->Arg(30)->Arg(50)->Arg(100);//->Arg(500);

static void stack_loop_only(benchmark::State &state) {
  const int N = state.range(0);
  nda::stack_array<long, 1, nda::static_extents(15)> A(N);
  nda::basic_array_view<long, 1, nda::C_layout, 'A', nda::no_alias_accessor, nda::borrowed> v{A};

  while (state.KeepRunning()) {
    for (int i = 0; i < N - 1; ++i) benchmark::DoNotOptimize(A(i) = i); //fnt(i));
    //for (int i = 0; i < N - 1; ++i) benchmark::DoNotOptimize(v(i) = i);//fnt(i));
  }
}
BENCHMARK(stack_loop_only)->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(10)->Arg(15); //->Arg(30)->Arg(50)->Arg(100);//->Arg(500);

// ------------------

using alloc_t = nda::mem::segregator<8 * 100, nda::mem::multi_bucket<8 * 100>, nda::mem::mallocator>;

static void mbucket_alloc(benchmark::State &state) {
  const int N = state.range(0);

  while (state.KeepRunning()) {
    nda::basic_array<long, 1, nda::C_layout, 'A', nda::heap_custom_alloc<alloc_t>> A(N);
    benchmark::DoNotOptimize(A(0));
  }
}
BENCHMARK(mbucket_alloc)->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(10)->Arg(15); //->Arg(30)->Arg(50)->Arg(100);//->Arg(500);

#if 0
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

#endif
