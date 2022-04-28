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

#include "./bench_common.hpp"
#include <nda/blas.hpp>

using value_t = double;

long Nmin = 64;
long Nmax = 1 << 13;

template <typename Vector>
static void DOT(benchmark::State &state) {
  long N = state.range(0);
  auto X = Vector{nda::rand<value_t>(N)};
  auto Y = Vector{nda::rand<value_t>(N)};
  for (auto _ : state) { nda::blas::dot(X, Y); }

  long NBytes = N * sizeof(value_t);
  //state.SetBytesProcessed(state.iterations() * NBytes);
  state.counters["bytesize"] = NBytes;
}
BENCHMARK_TEMPLATE(DOT, nda::vector<value_t>)->RangeMultiplier(2)->Range(Nmin, Nmax)->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(DOT, nda::cuvector<value_t>)->RangeMultiplier(2)->Range(Nmin, Nmax)->Unit(benchmark::kMicrosecond);

template <typename Matrix>
static void GEMM(benchmark::State &state) {
  long N = state.range(0);
  auto A = Matrix{nda::rand<value_t>(N, N)};
  auto B = Matrix{nda::rand<value_t>(N, N)};
  auto C = Matrix{nda::zeros<value_t>(N, N)};
  for (auto _ : state) { nda::blas::gemm(1.0, A, B, 0.0, C); }

  long NBytes = N * N * sizeof(value_t);
  //state.SetBytesProcessed(state.iterations() * NBytes);
  state.counters["bytesize"] = NBytes;
}
BENCHMARK_TEMPLATE(GEMM, nda::matrix<value_t>)->RangeMultiplier(2)->Range(Nmin, Nmax)->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(GEMM, nda::cumatrix<value_t>)->RangeMultiplier(2)->Range(Nmin, Nmax)->Unit(benchmark::kMicrosecond);

template <typename Vector, typename Matrix>
static void GER(benchmark::State &state) {
  long N = state.range(0);
  auto X = Vector{nda::rand<value_t>(N)};
  auto Y = Vector{nda::rand<value_t>(N)};
  auto M = Matrix{nda::zeros<value_t>(N, N)};
  for (auto _ : state) { nda::blas::ger(1.0, X, Y, M); }

  long NBytes = N * sizeof(value_t);
  //state.SetBytesProcessed(state.iterations() * NBytes);
  state.counters["bytesize"] = NBytes;
}
BENCHMARK_TEMPLATE(GER, nda::vector<value_t>, nda::matrix<value_t>)->RangeMultiplier(2)->Range(Nmin, Nmax)->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(GER, nda::cuvector<value_t>, nda::cumatrix<value_t>)->RangeMultiplier(2)->Range(Nmin, Nmax)->Unit(benchmark::kMicrosecond);
