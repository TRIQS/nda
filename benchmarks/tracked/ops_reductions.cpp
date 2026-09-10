// Copyright (c) 2020--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.


#include "./bench_ops.hpp"

using nda_bench::op_defaults;

struct op_sum : op_defaults {
  template <typename Arr> static decltype(auto) op(Arr const &A, Arr const &B) { return sum(A + B); }
};
struct op_prod : op_defaults {
  template <typename Arr> static decltype(auto) op(Arr const &A, Arr const &B) { return product(A + B); }
  // A + B must sit at ~1.0, or the running product overflows to +inf within a few hundred
  // elements (f32 by ~128 multiplications) and the benchmark times infinity arithmetic.
  static constexpr nda_bench::input_range inputs = {.scale = 1e-8, .offset = 0.5};
};
struct op_max_elem : op_defaults {
  template <typename Arr> static decltype(auto) op(Arr const &A, Arr const &B) { return max_element(A + B); }
  static constexpr bool supports_complex = false;
};
struct op_min_elem : op_defaults {
  template <typename Arr> static decltype(auto) op(Arr const &A, Arr const &B) { return min_element(A + B); }
  static constexpr bool supports_complex = false;
};
struct op_frob_norm : op_defaults {
  template <typename Arr> static decltype(auto) op(Arr const &A, Arr const &B) { return frobenius_norm(A + B); }
  static constexpr bool supports_rank3 = false;
};

NDA_BENCHMARK_ALL_TYPES(op_sum, "sum")
NDA_BENCHMARK_ALL_TYPES(op_prod, "product")
NDA_BENCHMARK_ALL_TYPES(op_max_elem, "max_element")
NDA_BENCHMARK_ALL_TYPES(op_min_elem, "min_element")
NDA_BENCHMARK_ALL_TYPES(op_frob_norm, "frobenius_norm")
