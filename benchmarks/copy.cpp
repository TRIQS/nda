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

using value_t = double;

template <size_t Rank>
using array_t = nda::array<value_t, Rank>;

template <size_t Rank>
using device_array_t = nda::cuarray<value_t, Rank>;

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
BENCHMARK_TEMPLATE(Copy, device_array_t<1>)->RangeMultiplier(8)->Range(KBmin, KBmax); // NOLINT

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
BENCHMARK_TEMPLATE(Copy1DStrided, device_array_t<1>)->RangeMultiplier(8)->Range(KBmin, KBmax); // NOLINT

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
BENCHMARK_TEMPLATE(CopyBlockStrided, device_array_t<2>, device_array_t<2>)->RangeMultiplier(8)->Range(KBmin, KBmax); // NOLINT
BENCHMARK_TEMPLATE(CopyBlockStrided, array_t<2>, device_array_t<2>)->RangeMultiplier(8)->Range(KBmin, KBmax);        // NOLINT
BENCHMARK_TEMPLATE(CopyBlockStrided, device_array_t<2>, array_t<2>)->RangeMultiplier(8)->Range(KBmin, KBmax);        // NOLINT
