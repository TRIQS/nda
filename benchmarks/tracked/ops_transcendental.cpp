// Copyright (c) 2020--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#include "./bench_ops.hpp"
using namespace nda_bench;

struct op_no_matrix : op_defaults {
  static constexpr bool supports_matrix = false;
};

struct op_exp : op_no_matrix {
  static decltype(auto) op(auto const &A) { return exp(A); }
};
struct op_log : op_no_matrix {
  static decltype(auto) op(auto const &A) { return log(A); }
};
struct op_sqrt : op_no_matrix {
  static decltype(auto) op(auto const &A) { return sqrt(A); }
};
struct op_sin : op_no_matrix {
  static decltype(auto) op(auto const &A) { return sin(A); }
};
struct op_cos : op_no_matrix {
  static decltype(auto) op(auto const &A) { return cos(A); }
};
struct op_tan : op_no_matrix {
  static decltype(auto) op(auto const &A) { return tan(A); }
};
struct op_sinh : op_no_matrix {
  static decltype(auto) op(auto const &A) { return sinh(A); }
};
struct op_cosh : op_no_matrix {
  static decltype(auto) op(auto const &A) { return cosh(A); }
};
struct op_tanh : op_no_matrix {
  static decltype(auto) op(auto const &A) { return tanh(A); }
};
struct op_asin : op_no_matrix {
  static decltype(auto) op(auto const &A) { return asin(A); }
};
struct op_acos : op_no_matrix {
  static decltype(auto) op(auto const &A) { return acos(A); }
};
struct op_atan : op_no_matrix {
  static decltype(auto) op(auto const &A) { return atan(A); }
};

NDA_BENCHMARK(op_exp, "exp", types<double, std::complex<double>>, array_input<2>)

NDA_BENCHMARK(op_log, "log", types<double, std::complex<double>>, array_input<2>)

NDA_BENCHMARK(op_sqrt, "sqrt", types<double, std::complex<double>>, array_input<2>)

NDA_BENCHMARK(op_sin, "sin", types<double, std::complex<double>>, array_input<2>)

NDA_BENCHMARK(op_cos, "cos", types<double, std::complex<double>>, array_input<2>)

NDA_BENCHMARK(op_tan, "tan", types<double, std::complex<double>>, array_input<2>)

NDA_BENCHMARK(op_sinh, "sinh", types<double, std::complex<double>>, array_input<2>)

NDA_BENCHMARK(op_cosh, "cosh", types<double, std::complex<double>>, array_input<2>)

NDA_BENCHMARK(op_tanh, "tanh", types<double, std::complex<double>>, array_input<2>)

// domain is [-1,1]: the default [1,2) range would return NaN for every element
NDA_BENCHMARK(op_asin, "asin", types<double, std::complex<double>>, array_input<2, 'A', nda::C_layout, nda_bench::signed_band>)

// domain is [-1,1]: the default [1,2) range would return NaN for every element
NDA_BENCHMARK(op_acos, "acos", types<double, std::complex<double>>, array_input<2, 'A', nda::C_layout, nda_bench::signed_band>)

NDA_BENCHMARK(op_atan, "atan", types<double, std::complex<double>>, array_input<2>)
