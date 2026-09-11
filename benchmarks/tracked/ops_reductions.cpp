// Copyright (c) 2020--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#include "./bench_ops.hpp"

using nda_bench::op_defaults;

struct op_sum : op_defaults {
  static decltype(auto) op(auto const &A, auto const &B) { return sum(A + B); }
};
struct op_prod : op_defaults {
  static decltype(auto) op(auto const &A, auto const &B) { return product(A + B); }
};
struct op_max_elem : op_defaults {
  static decltype(auto) op(auto const &A, auto const &B) { return max_element(A + B); }
  static constexpr bool supports_complex = false;
};
struct op_min_elem : op_defaults {
  static decltype(auto) op(auto const &A, auto const &B) { return min_element(A + B); }
  static constexpr bool supports_complex = false;
};
struct op_frob_norm : op_defaults {
  static decltype(auto) op(auto const &A, auto const &B) { return frobenius_norm(A + B); }
  static constexpr bool supports_rank3 = false;
};

using namespace nda_bench;

NDA_BENCHMARK(op_sum, "sum", array_input<2>, array_input<2>)
NDA_BENCHMARK(op_sum, "sum", matrix_input<>, matrix_input<>)

// A + B must sit at ~1.0, or the running product overflows to +inf within a few hundred
// elements (f32 by ~128 multiplications) and the benchmark times infinity arithmetic.
NDA_BENCHMARK(op_prod, "product", array_input<2, 'A', nda::C_layout, nda_bench::product_band>,
              array_input<2, 'A', nda::C_layout, nda_bench::product_band>)
NDA_BENCHMARK(op_prod, "product", matrix_input<nda::C_layout, nda_bench::product_band>, matrix_input<nda::C_layout, nda_bench::product_band>)

NDA_BENCHMARK(op_max_elem, "max_element", array_input<2>, array_input<2>)

NDA_BENCHMARK(op_min_elem, "min_element", array_input<2>, array_input<2>)

NDA_BENCHMARK(op_frob_norm, "frobenius_norm", array_input<2>, array_input<2>)
