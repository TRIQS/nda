// Copyright (c) 2022--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#include "./bench_common.hpp"
#include <nda/blas.hpp>

using value_t = double;

const long Nmin = 32;
const long Nmax = 1 << 9;

template <typename M>
static void GEMM_BATCH(benchmark::State &state) {
  long N          = state.range(0);
  long BatchCount = 10 * Nmax * Nmax / N / N;

  auto A = std::vector(BatchCount, M{nda::rand<value_t>(N, N)});
  auto B = std::vector(BatchCount, M{nda::rand<value_t>(N, N)});
  auto C = std::vector(BatchCount, M{nda::zeros<value_t>(N, N)});

  for (auto s : state) { nda::blas::gemm_batch(1.0, A, B, 0.0, C); }

  auto NBytes                  = BatchCount * N * N * sizeof(value_t);
  state.counters["batchcount"] = double(BatchCount);
  state.counters["bytesize"]   = double(NBytes);
}
BENCHMARK_TEMPLATE(GEMM_BATCH, nda::matrix<value_t>)->RangeMultiplier(2)->Range(Nmin, Nmax)->Unit(benchmark::kMicrosecond);   // NOLINT
#ifdef NDA_HAVE_CUDA
BENCHMARK_TEMPLATE(GEMM_BATCH, nda::cumatrix<value_t>)->RangeMultiplier(2)->Range(Nmin, Nmax)->Unit(benchmark::kMicrosecond); // NOLINT
#endif

template <typename M>
static void GEMM_VBATCH(benchmark::State &state) {
  long N          = state.range(0);
  long BatchCount = 10 * Nmax * Nmax / N / N;

  auto A = std::vector(BatchCount, M{nda::rand<value_t>(N, N)});
  auto B = std::vector(BatchCount, M{nda::rand<value_t>(N, N)});
  auto C = std::vector(BatchCount, M{nda::zeros<value_t>(N, N)});

  for (auto s : state) { nda::blas::gemm_vbatch(1.0, A, B, 0.0, C); }

  auto NBytes                  = BatchCount * N * N * sizeof(value_t);
  state.counters["batchcount"] = double(BatchCount);
  state.counters["bytesize"]   = double(NBytes);
}
BENCHMARK_TEMPLATE(GEMM_VBATCH, nda::matrix<value_t>)->RangeMultiplier(2)->Range(Nmin, Nmax)->Unit(benchmark::kMicrosecond);   // NOLINT
#ifdef NDA_HAVE_CUDA
BENCHMARK_TEMPLATE(GEMM_VBATCH, nda::cumatrix<value_t>)->RangeMultiplier(2)->Range(Nmin, Nmax)->Unit(benchmark::kMicrosecond); // NOLINT
#endif

template <typename T>
static void GEMM_BATCH_STRIDED(benchmark::State &state) {
  long N          = state.range(0);
  long BatchCount = 10 * Nmax * Nmax / N / N;

  auto A = T{nda::rand<value_t>(BatchCount, N, N)};
  auto B = T{nda::rand<value_t>(BatchCount, N, N)};
  auto C = T{nda::zeros<value_t>(BatchCount, N, N)};

  for (auto s : state) { nda::blas::gemm_batch_strided(1.0, A, B, 0.0, C); }

  auto NBytes                  = BatchCount * N * N * sizeof(value_t);
  state.counters["batchcount"] = double(BatchCount);
  state.counters["bytesize"]   = double(NBytes);
}
BENCHMARK_TEMPLATE(GEMM_BATCH_STRIDED, nda::array<value_t, 3>)->RangeMultiplier(2)->Range(Nmin, Nmax)->Unit(benchmark::kMicrosecond);   // NOLINT
#ifdef NDA_HAVE_CUDA
BENCHMARK_TEMPLATE(GEMM_BATCH_STRIDED, nda::cuarray<value_t, 3>)->RangeMultiplier(2)->Range(Nmin, Nmax)->Unit(benchmark::kMicrosecond); // NOLINT
#endif
