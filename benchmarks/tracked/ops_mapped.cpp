// Copyright (c) 2020--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#include "./bench_ops.hpp"

using nda_bench::op_defaults;

struct op_pow : op_defaults {
  static decltype(auto) op(auto const &A) { return pow(A, 3.5); }
};
struct op_conj : op_defaults {
  static decltype(auto) op(auto const &A) { return conj(A); }
};
struct op_abs : op_defaults {
  static decltype(auto) op(auto const &A) { return abs(A); }
};
struct op_imag : op_defaults {
  static decltype(auto) op(auto const &A) { return imag(A); }
};
struct op_real : op_defaults {
  static decltype(auto) op(auto const &A) { return real(A); }
};
struct op_abs2 : op_defaults {
  static decltype(auto) op(auto const &A) { return abs2(A); }
};
struct op_floor : op_defaults {
  static decltype(auto) op(auto const &A) { return floor(A); }
  static constexpr bool supports_complex = false;
};
struct op_isnan : op_defaults {
  static decltype(auto) op(auto const &A) { return isnan(A); }
};
struct op_max : op_defaults {
  static decltype(auto) op(auto const &A, auto const &B) { return max(A, B); }
  static constexpr bool supports_complex = false;
};
struct op_min : op_defaults {
  static decltype(auto) op(auto const &A, auto const &B) { return min(A, B); }
  static constexpr bool supports_complex = false;
};

using namespace nda_bench;

NDA_BENCHMARK(op_pow, "pow", types<double, std::complex<double>>, array_input<2>)

NDA_BENCHMARK(op_conj, "conj", types<double, std::complex<double>>, array_input<2>)

NDA_BENCHMARK(op_abs, "abs", types<double, std::complex<double>>, array_input<2>)

NDA_BENCHMARK(op_imag, "imag", types<std::complex<double>>, array_input<2>)

NDA_BENCHMARK(op_real, "real", types<std::complex<double>>, array_input<2>)

NDA_BENCHMARK(op_abs2, "abs2", types<double, std::complex<double>>, array_input<2>)

NDA_BENCHMARK(op_floor, "floor", types<double>, array_input<2>)

NDA_BENCHMARK(op_isnan, "isnan", types<double, std::complex<double>>, array_input<2>)

NDA_BENCHMARK(op_max, "max", types<double>, array_input<2>, array_input<2>)

NDA_BENCHMARK(op_min, "min", types<double>, array_input<2>, array_input<2>)
