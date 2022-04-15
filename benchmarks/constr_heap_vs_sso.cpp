// Copyright (c) 2019-2020 Simons Foundation
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

#include <benchmark/benchmark.h>
#include <nda/nda.hpp>

using namespace nda;

#define BENCH_EXPR(NAME, EXPR)                                                                                                                       \
  static void NAME(benchmark::State &state) {                                                                                                        \
    for (auto _ : state) { benchmark::DoNotOptimize(EXPR); }                                                                                         \
  }                                                                                                                                                  \
  BENCHMARK(NAME);

// --- Matrix on the heap

using mat_heap_t = matrix<double>;
BENCH_EXPR(constr_mat_heap, mat_heap_t{});
BENCH_EXPR(constr_vec_heap_mat_heap, vector<mat_heap_t>(1));
BENCH_EXPR(constr_vec_sso_mat_heap, (vector<mat_heap_t, sso<10>>(1)));

// --- SSO Matrix

using mat_sso_t = matrix<double, C_layout, sso<1000>>;
BENCH_EXPR(constr_mat_sso, mat_sso_t{});
BENCH_EXPR(constr_vec_heap_mat_sso, std::vector<mat_sso_t>(1));
BENCH_EXPR(constr_vec_sso_mat_sso, (vector<mat_sso_t, sso<10>>(1)));

// --- Heap vs SSO Handle

BENCH_EXPR(constr_handle_heap, heap<>::handle<double>{});
BENCH_EXPR(constr_handle_sso, sso<1000>::handle<double>{});

// --- Value Initialization

struct toy_mat1_t {
  sso<1000>::handle<double> storage;
  toy_mat1_t() = default; // Aggregate type -> Zero init of SSO buffer
};

struct toy_mat2_t {
  sso<1000>::handle<double> storage;
  toy_mat2_t(){}; // Custom constructor -> Not an aggregate type
};

BENCH_EXPR(constr_toy_mat1, toy_mat1_t{});
BENCH_EXPR(constr_toy_mat2, toy_mat2_t{});
