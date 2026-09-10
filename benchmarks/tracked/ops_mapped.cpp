// Copyright (c) 2020--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.


#include "./bench_ops.hpp"

using nda_bench::op_defaults;

struct op_pow : op_defaults {
  template <typename Arr> static decltype(auto) op(Arr const &A) { return pow(A, 3.5); }
};
struct op_conj : op_defaults {
  template <typename Arr> static decltype(auto) op(Arr const &A) { return conj(A); }
};
struct op_abs : op_defaults {
  template <typename Arr> static decltype(auto) op(Arr const &A) { return abs(A); }
};
struct op_imag : op_defaults {
  template <typename Arr> static decltype(auto) op(Arr const &A) { return imag(A); }
};
struct op_real : op_defaults {
  template <typename Arr> static decltype(auto) op(Arr const &A) { return real(A); }
};
struct op_abs2 : op_defaults {
  template <typename Arr> static decltype(auto) op(Arr const &A) { return abs2(A); }
};
struct op_floor : op_defaults {
  template <typename Arr> static decltype(auto) op(Arr const &A) { return floor(A); }
  static constexpr bool supports_complex = false;
};
struct op_isnan : op_defaults {
  template <typename Arr> static decltype(auto) op(Arr const &A) { return isnan(A); }
};
struct op_max : op_defaults {
  template <typename Arr> static decltype(auto) op(Arr const &A, Arr const &B) { return max(A, B); }
  static constexpr bool supports_complex = false;
};
struct op_min : op_defaults {
  template <typename Arr> static decltype(auto) op(Arr const &A, Arr const &B) { return min(A, B); }
  static constexpr bool supports_complex = false;
};

NDA_BENCHMARK_ALL_TYPES(op_pow, "pow")
NDA_BENCHMARK_ALL_TYPES(op_conj, "conj")
NDA_BENCHMARK_ALL_TYPES(op_abs, "abs")
NDA_BENCHMARK_ALL_TYPES(op_imag, "imag")
NDA_BENCHMARK_ALL_TYPES(op_real, "real")
NDA_BENCHMARK_ALL_TYPES(op_abs2, "abs2")
NDA_BENCHMARK_ALL_TYPES(op_floor, "floor")
NDA_BENCHMARK_ALL_TYPES(op_isnan, "isnan")
NDA_BENCHMARK_ALL_TYPES(op_max, "max")
NDA_BENCHMARK_ALL_TYPES(op_min, "min")
