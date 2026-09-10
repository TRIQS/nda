// Copyright (c) 2020--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

// Arithmetic operators and fused expressions. Note that for nda::matrix (algebra 'M') the
// operators differ from nda::array: A * B is linalg::matmul and A / B is inv(B) then gemm,
// both eager. So the m2 series of fma is a gemm into a temporary followed by an add, which
// is the expression worth folding into a single gemm with beta = 1.

#include "./bench_ops.hpp"

using nda_bench::op_defaults;

struct op_neg : op_defaults {
  template <typename Arr> static decltype(auto) op(Arr const &A) { return -A; }
};
struct op_add : op_defaults {
  template <typename Arr> static decltype(auto) op(Arr const &A, Arr const &B) { return A + B; }
};
struct op_sub : op_defaults {
  template <typename Arr> static decltype(auto) op(Arr const &A, Arr const &B) { return A - B; }
};
struct op_mul : op_defaults {
  template <typename Arr> static decltype(auto) op(Arr const &A, Arr const &B) { return A * B; }
};
struct op_div : op_defaults {
  template <typename Arr> static decltype(auto) op(Arr const &A, Arr const &B) { return A / B; }
};
// Elementwise product like mul on a2/a3, but routed through map() rather than expr; on m2
// it stays elementwise while mul dispatches to gemm.
struct op_hadamard : op_defaults {
  template <typename Arr> static decltype(auto) op(Arr const &A, Arr const &B) { return hadamard(A, B); }
};
struct op_fma : op_defaults {
  template <typename Arr> static decltype(auto) op(Arr const &A, Arr const &B, Arr const &C) { return A * B + C; }
};
struct op_fms : op_defaults {
  template <typename Arr> static decltype(auto) op(Arr const &A, Arr const &B, Arr const &C) { return A * B - C; }
};
struct op_addsub : op_defaults {
  template <typename Arr> static decltype(auto) op(Arr const &A, Arr const &B, Arr const &C) { return A + B - C; }
};

NDA_BENCHMARK_ALL_TYPES(op_neg, "neg")
NDA_BENCHMARK_ALL_TYPES(op_add, "add")
NDA_BENCHMARK_ALL_TYPES(op_sub, "sub")
NDA_BENCHMARK_ALL_TYPES(op_mul, "mul")
NDA_BENCHMARK_ALL_TYPES(op_div, "div")
NDA_BENCHMARK_ALL_TYPES(op_hadamard, "hadamard")
NDA_BENCHMARK_ALL_TYPES(op_fma, "fma")
NDA_BENCHMARK_ALL_TYPES(op_fms, "fms")
NDA_BENCHMARK_ALL_TYPES(op_addsub, "addsub")
