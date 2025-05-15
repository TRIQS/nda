// Copyright (c) 2022--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#include "./bench_common.hpp"

template <int N>
static void inv(benchmark::State &state) {

  nda::matrix<double> W(N, N), Wi(N, N);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) W(i, j) = (i > j ? 0.5 + i + 2.5 * j : i * 0.8 - j - 0.5);

  while (state.KeepRunning()) { benchmark::DoNotOptimize(Wi = nda::linalg::inv(W)); }
}

BENCHMARK_TEMPLATE(inv, 1);
BENCHMARK_TEMPLATE(inv, 2);
BENCHMARK_TEMPLATE(inv, 3);
