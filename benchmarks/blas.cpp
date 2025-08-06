// Copyright (c) 2022--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#include "./bench_common.hpp"
#include <nda/blas.hpp>

using value_t = double;

const long Nmin = 64;
const long Nmax = 1 << 13;

template <typename Vector>
static void DOT(benchmark::State &state) {
  long N = state.range(0);
  auto X = Vector{nda::rand<value_t>(N)};
  auto Y = Vector{nda::rand<value_t>(N)};
  for (auto s : state) { nda::blas::dot(X, Y); }

  auto NBytes                = N * sizeof(value_t);
  state.counters["bytesize"] = double(NBytes);
}
BENCHMARK_TEMPLATE(DOT, nda::vector<value_t>)->RangeMultiplier(2)->Range(Nmin, Nmax)->Unit(benchmark::kMicrosecond);   // NOLINT
#ifdef NDA_HAVE_CUDA
BENCHMARK_TEMPLATE(DOT, nda::cuvector<value_t>)->RangeMultiplier(2)->Range(Nmin, Nmax)->Unit(benchmark::kMicrosecond); // NOLINT
#endif

template <typename Matrix>
static void GEMM(benchmark::State &state) {
  long N = state.range(0);
  auto A = Matrix{nda::rand<value_t>(N, N)};
  auto B = Matrix{nda::rand<value_t>(N, N)};
  auto C = Matrix{nda::zeros<value_t>(N, N)};
  for (auto s : state) { nda::blas::gemm(1.0, A, B, 0.0, C); }

  auto NBytes                = N * N * sizeof(value_t);
  state.counters["bytesize"] = double(NBytes);
}
BENCHMARK_TEMPLATE(GEMM, nda::matrix<value_t>)->RangeMultiplier(2)->Range(Nmin, Nmax)->Unit(benchmark::kMicrosecond);   // NOLINT
#ifdef NDA_HAVE_CUDA
BENCHMARK_TEMPLATE(GEMM, nda::cumatrix<value_t>)->RangeMultiplier(2)->Range(Nmin, Nmax)->Unit(benchmark::kMicrosecond); // NOLINT
#endif

template <typename Vector, typename Matrix>
static void GER(benchmark::State &state) {
  long N = state.range(0);
  auto X = Vector{nda::rand<value_t>(N)};
  auto Y = Vector{nda::rand<value_t>(N)};
  auto M = Matrix{nda::zeros<value_t>(N, N)};
  for (auto s : state) { nda::blas::ger(1.0, X, Y, M); }

  auto NBytes                = N * sizeof(value_t);
  state.counters["bytesize"] = double(NBytes);
}
BENCHMARK_TEMPLATE(GER, nda::vector<value_t>, nda::matrix<value_t>)->RangeMultiplier(2)->Range(Nmin, Nmax)->Unit(benchmark::kMicrosecond); // NOLINT
#ifdef NDA_HAVE_CUDA
BENCHMARK_TEMPLATE(GER, nda::cuvector<value_t>, nda::cumatrix<value_t>)
   ->RangeMultiplier(2)
   ->Range(Nmin, Nmax)
   ->Unit(benchmark::kMicrosecond); // NOLINT
#endif
