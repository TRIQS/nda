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

static constexpr int N1 = 4, N2 = 4;

// ------------------------------- 2d ----------------------------------------

static void for2(benchmark::State &state) {
  nda::array<double, 2> a(N1, N2);
  const long l0 = a.indexmap().lengths()[0];
  const long l1 = a.indexmap().lengths()[1];

  while (state.KeepRunning()) {
    for (long i = 0; i < l0; ++i)
      for (long j = 0; j < l1; ++j) { benchmark::DoNotOptimize(a(i, j) = 10); }
  }
}
BENCHMARK(for2);

static void foreach2(benchmark::State &state) {
  nda::array<double, 2> a(N1, N2);

  while (state.KeepRunning()) {
    nda::for_each(a.shape(), [&a](auto x0, auto x1) { benchmark::DoNotOptimize(a(x0, x1) = 10); });
  }
}
BENCHMARK(foreach2);

static void foreach_static2(benchmark::State &state) {
  nda::array<double, 2> a(N1, N2);

  while (state.KeepRunning()) {
    nda::for_each_static<encode(std::array{N1, N2}), 0>(a.shape(), [&a](auto x0, auto x1) { benchmark::DoNotOptimize(a(x0, x1) = 10); });
  }
}
BENCHMARK(foreach_static2);
