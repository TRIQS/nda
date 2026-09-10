// Copyright (c) 2020--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.


#include "./bench_ops.hpp"

struct op_no_matrix : nda_bench::op_defaults {
  static constexpr bool supports_matrix = false;
};

struct op_exp : op_no_matrix {
  template <typename Arr> static decltype(auto) op(Arr const &A) { return exp(A); }
};
struct op_log : op_no_matrix {
  template <typename Arr> static decltype(auto) op(Arr const &A) { return log(A); }
};
struct op_sqrt : op_no_matrix {
  template <typename Arr> static decltype(auto) op(Arr const &A) { return sqrt(A); }
};
struct op_sin : op_no_matrix {
  template <typename Arr> static decltype(auto) op(Arr const &A) { return sin(A); }
};
struct op_cos : op_no_matrix {
  template <typename Arr> static decltype(auto) op(Arr const &A) { return cos(A); }
};
struct op_tan : op_no_matrix {
  template <typename Arr> static decltype(auto) op(Arr const &A) { return tan(A); }
};
struct op_sinh : op_no_matrix {
  template <typename Arr> static decltype(auto) op(Arr const &A) { return sinh(A); }
};
struct op_cosh : op_no_matrix {
  template <typename Arr> static decltype(auto) op(Arr const &A) { return cosh(A); }
};
struct op_tanh : op_no_matrix {
  template <typename Arr> static decltype(auto) op(Arr const &A) { return tanh(A); }
};
struct op_asin : op_no_matrix {
  template <typename Arr> static decltype(auto) op(Arr const &A) { return asin(A); }
  // domain is [-1,1]: the default [1,2) range would return NaN for every element
  static constexpr nda_bench::input_range inputs = {.scale = 2.0, .offset = -1.0};
};
struct op_acos : op_no_matrix {
  template <typename Arr> static decltype(auto) op(Arr const &A) { return acos(A); }
  // domain is [-1,1]: the default [1,2) range would return NaN for every element
  static constexpr nda_bench::input_range inputs = {.scale = 2.0, .offset = -1.0};
};
struct op_atan : op_no_matrix {
  template <typename Arr> static decltype(auto) op(Arr const &A) { return atan(A); }
};

NDA_BENCHMARK_ALL_TYPES(op_exp, "exp")
NDA_BENCHMARK_ALL_TYPES(op_log, "log")
NDA_BENCHMARK_ALL_TYPES(op_sqrt, "sqrt")
NDA_BENCHMARK_ALL_TYPES(op_sin, "sin")
NDA_BENCHMARK_ALL_TYPES(op_cos, "cos")
NDA_BENCHMARK_ALL_TYPES(op_tan, "tan")
NDA_BENCHMARK_ALL_TYPES(op_sinh, "sinh")
NDA_BENCHMARK_ALL_TYPES(op_cosh, "cosh")
NDA_BENCHMARK_ALL_TYPES(op_tanh, "tanh")
NDA_BENCHMARK_ALL_TYPES(op_asin, "asin")
NDA_BENCHMARK_ALL_TYPES(op_acos, "acos")
NDA_BENCHMARK_ALL_TYPES(op_atan, "atan")
