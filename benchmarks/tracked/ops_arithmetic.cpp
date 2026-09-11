// Copyright (c) 2020--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

// Arithmetic operators and fused expressions. Note that for nda::matrix (algebra 'M') the
// operators differ from nda::array: A * B is linalg::matmul and A / B is inv(B) then gemm,
// both eager. So the m2 series of fma is a gemm into a temporary followed by an add, which
// is the expression worth folding into a single gemm with beta = 1.

#include "./bench_ops.hpp"

using namespace nda_bench;

struct op_neg : op_defaults {
  static decltype(auto) op(auto const &A) { return -A; }
};
struct op_add : op_defaults {
  static decltype(auto) op(auto const &A, auto const &B) { return A + B; }
};
struct op_sub : op_defaults {
  static decltype(auto) op(auto const &A, auto const &B) { return A - B; }
};
struct op_mul : op_defaults {
  static auto result_shape(auto const &A, auto const &B) { return nda_bench::product_shape(A, B); }
  static decltype(auto) op(auto const &A, auto const &B) { return A * B; }
};
struct op_div : op_defaults {
  static auto result_shape(auto const &A, auto const &B) { return nda_bench::product_shape(A, B); }
  static decltype(auto) op(auto const &A, auto const &B) { return A / B; }
};
struct op_hadamard : op_defaults {
  static decltype(auto) op(auto const &A, auto const &B) { return hadamard(A, B); }
};
struct op_fma : op_defaults {
  static auto result_shape(auto const &A, auto const &B, auto const &C) {
    if constexpr (nda::Scalar<std::remove_cvref_t<decltype(A)>> && nda::Scalar<std::remove_cvref_t<decltype(B)>>) {
      return C.shape();
    } else {
      return nda_bench::product_shape(A, B);
    }
  }
  static decltype(auto) op(auto const &A, auto const &B, auto const &C) { return A * B + C; }
};
struct op_fms : op_defaults {
  static auto result_shape(auto const &A, auto const &B, auto const &C) {
    if constexpr (nda::Scalar<std::remove_cvref_t<decltype(A)>> && nda::Scalar<std::remove_cvref_t<decltype(B)>>) {
      return C.shape();
    } else {
      return nda_bench::product_shape(A, B);
    }
  }
  static decltype(auto) op(auto const &A, auto const &B, auto const &C) { return A * B - C; }
};
struct op_addsub : op_defaults {
  static decltype(auto) op(auto const &A, auto const &B, auto const &C) { return A + B - C; }
};

NDA_BENCHMARK(op_neg, "neg", array_input<2>)

NDA_BENCHMARK(op_add, "add", array_input<2>, array_input<2>)
NDA_BENCHMARK(op_add, "add", matrix_input<>, scalar_input<>)
NDA_BENCHMARK(op_add, "add", array_input<2>, scalar_input<>)
NDA_BENCHMARK(op_add, "add", row_slice<matrix_input<>>, row_slice<matrix_input<>>)
NDA_BENCHMARK(op_add, "add", col_slice<matrix_input<>>, col_slice<matrix_input<>>)
NDA_BENCHMARK(op_add, "add", row_slice<array_input<2>>, scalar_input<>)
NDA_BENCHMARK(op_add, "add", vector_input<>, vector_input<>)

NDA_BENCHMARK(op_sub, "sub", array_input<2>, array_input<2>)
NDA_BENCHMARK(op_sub, "sub", vector_input<>, vector_input<>)

NDA_BENCHMARK(op_mul, "mul", array_input<2>, array_input<2>)
NDA_BENCHMARK(op_mul, "mul", matrix_input<>, matrix_input<>)
NDA_BENCHMARK(op_mul, "mul", matrix_input<>, vector_input<>)
NDA_BENCHMARK(op_mul, "mul", vector_input<>, scalar_input<>)
NDA_BENCHMARK(op_mul, "mul", matrix_input<>, scalar_input<>)

NDA_BENCHMARK(op_div, "div", array_input<2, 'A', nda::C_layout, signed_band>, array_input<2, 'A', nda::C_layout, positive_band>)
NDA_BENCHMARK(op_div, "div", matrix_input<>, matrix_input<>)
NDA_BENCHMARK(op_div, "div", vector_input<>, scalar_input<>)
NDA_BENCHMARK(op_div, "div", matrix_input<>, scalar_input<>)

NDA_BENCHMARK(op_hadamard, "hadamard", array_input<2>, array_input<2>)

NDA_BENCHMARK(op_fma, "fma", array_input<2>, array_input<2>, array_input<2>)
NDA_BENCHMARK(op_fma, "fma", matrix_input<>, matrix_input<>, matrix_input<>)
NDA_BENCHMARK(op_fma, "fma", matrix_input<>, scalar_input<>, matrix_input<>)
NDA_BENCHMARK(op_fma, "fma", matrix_input<>, vector_input<>, vector_input<>)
NDA_BENCHMARK(op_fma, "fma", scalar_input<>, vector_input<>, vector_input<>)

NDA_BENCHMARK(op_fms, "fms", array_input<2>, array_input<2>, array_input<2>)
NDA_BENCHMARK(op_fms, "fms", matrix_input<>, matrix_input<>, matrix_input<>)
NDA_BENCHMARK(op_fms, "fms", matrix_input<>, vector_input<>, vector_input<>)

NDA_BENCHMARK(op_addsub, "addsub", array_input<2>, array_input<2>, array_input<2>)
