// Copyright (c) 2022--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#include "./bench_common.hpp"

using value_t = double;

template <size_t Rank>
using array_t = nda::array<value_t, Rank>;

#ifdef NDA_HAVE_CUDA
template <size_t Rank>
using device_array_t = nda::cuarray<value_t, Rank>;
#endif

const long KBmin = 8;
const long KBmax = 1 << 15;

template <typename Array>
static void Copy(benchmark::State &state) {
  long NBytes = state.range(0) * 1024;
  long N      = NBytes / sizeof(value_t);
  auto src    = Array{nda::rand<value_t>(N)};
  auto dst    = Array{nda::zeros<value_t>(N)};
  for (auto s : state) { dst = src; }
  state.SetBytesProcessed(state.iterations() * NBytes);
  state.counters["processed"] = double(NBytes);
}
BENCHMARK_TEMPLATE(Copy, array_t<1>)->RangeMultiplier(8)->Range(KBmin, KBmax);        // NOLINT
#ifdef NDA_HAVE_CUDA
BENCHMARK_TEMPLATE(Copy, device_array_t<1>)->RangeMultiplier(8)->Range(KBmin, KBmax); // NOLINT
#endif
template <typename Array>
static void Copy1DStrided(benchmark::State &state) {
  long NBytes = state.range(0) * 1024;
  long step   = 10;
  long N      = step * NBytes / sizeof(value_t);
  auto src    = Array{nda::rand<value_t>(N)};
  auto dst    = Array{nda::zeros<value_t>(N)};
  auto src_v  = src(range(0, N, step));
  auto dst_v  = dst(range(0, N, step));
  for (auto s : state) { dst_v = src_v; }
  state.SetBytesProcessed(state.iterations() * NBytes);
  state.counters["processed"] = double(NBytes);
  state.counters["step"]      = double(step);
}
BENCHMARK_TEMPLATE(Copy1DStrided, array_t<1>)->RangeMultiplier(8)->Range(KBmin, KBmax);        // NOLINT
#ifdef NDA_HAVE_CUDA
BENCHMARK_TEMPLATE(Copy1DStrided, device_array_t<1>)->RangeMultiplier(8)->Range(KBmin, KBmax); // NOLINT
#endif

template <typename DstArray, typename SrcArray>
static void CopyBlockStrided(benchmark::State &state) {
  long NBytesPerBlock = state.range(0) * 1024;
  long step           = 2;
  long n_blocks       = 10;
  long N              = NBytesPerBlock / sizeof(value_t);
  auto src            = SrcArray{nda::rand<value_t>(step * n_blocks, N)};
  auto dst            = DstArray{nda::zeros<value_t>(step * n_blocks, N)};
  auto src_v          = src(range(0, step * n_blocks, step), _);
  auto dst_v          = dst(range(0, step * n_blocks, step), _);
  for (auto s : state) { dst_v = src_v; }
  state.SetBytesProcessed(state.iterations() * NBytesPerBlock * n_blocks);
  state.counters["processed"] = double(NBytesPerBlock * n_blocks);
  state.counters["step"]      = double(step);
  state.counters["n_blocks"]  = double(n_blocks);
}
BENCHMARK_TEMPLATE(CopyBlockStrided, array_t<2>, array_t<2>)->RangeMultiplier(8)->Range(KBmin, KBmax);               // NOLINT
#ifdef NDA_HAVE_CUDA
BENCHMARK_TEMPLATE(CopyBlockStrided, device_array_t<2>, device_array_t<2>)->RangeMultiplier(8)->Range(KBmin, KBmax); // NOLINT
BENCHMARK_TEMPLATE(CopyBlockStrided, array_t<2>, device_array_t<2>)->RangeMultiplier(8)->Range(KBmin, KBmax);        // NOLINT
BENCHMARK_TEMPLATE(CopyBlockStrided, device_array_t<2>, array_t<2>)->RangeMultiplier(8)->Range(KBmin, KBmax);        // NOLINT
#endif
