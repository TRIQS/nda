// Copyright (c) 2020--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

// In-place nda::tensor ops: scalars and input bands keep repeated calls bounded.

#include "./bench_ops.hpp"

using nda_bench::op_defaults;
using nda::tensor::binary_op;
using nda::tensor::unary_op;

template <typename A>
using value_t = nda::get_value_t<std::remove_cvref_t<A>>;

struct log_band {
  static constexpr nda_bench::input_range value{.lower = 3.0, .upper = 6.0};
};

// a = alpha * op(a)
template <unary_op Op, double Alpha>
struct op_scale : op_defaults {
  static constexpr std::size_t target = 0;
  static void op(auto &out, auto const &) { nda::tensor::scale(value_t<decltype(out)>(Alpha), out, Op); }
};

// b = op(alpha * a, beta * b)
template <binary_op Op, double Alpha, double Beta, bool Complex = true>
struct op_elementwise : op_defaults {
  static constexpr std::size_t target    = 1;
  static constexpr bool supports_complex = Complex;
  static void op(auto &out, auto const &A, auto const &) {
    nda::tensor::elementwise(value_t<decltype(A)>(Alpha), A, value_t<decltype(A)>(Beta), out, Op);
  }
};

// c = op_ABC(op_AB(alpha * a, beta * b), gamma * c)
template <binary_op OpAB, binary_op OpABC, double Alpha, double Beta, double Gamma, bool Complex = true>
struct op_elementwise_trinary : op_defaults {
  static constexpr std::size_t target    = 2;
  static constexpr bool supports_complex = Complex;
  static void op(auto &out, auto const &A, auto const &B, auto const &) {
    using T          = value_t<decltype(A)>;
    auto const index = nda::tensor::default_index<nda::get_rank<std::remove_cvref_t<decltype(A)>>>();
    nda::tensor::elementwise_trinary(T(Alpha), A, index, T(Beta), B, index, T(Gamma), out, index, OpAB, OpABC);
  }
};

using op_scale_identity = op_scale<unary_op::IDENTITY, -1.0>;
using op_scale_conj     = op_scale<unary_op::CONJ, -1.0>;
using op_scale_abs      = op_scale<unary_op::ABS, 1.0>;
using op_scale_sqrt     = op_scale<unary_op::SQRT, 1.0>;
using op_scale_exp      = op_scale<unary_op::EXP, 0.25>;
using op_scale_log      = op_scale<unary_op::LOG, 3.0>;
using op_scale_rcp      = op_scale<unary_op::RCP, 1.0>;

using op_ew_sum     = op_elementwise<binary_op::SUM, 1.0, 0.5>;
using op_ew_prod    = op_elementwise<binary_op::PROD, 2.0, 1.0>;
using op_ew_sum_abs = op_elementwise<binary_op::SUM_ABS, 1.0, 0.5>;
using op_ew_max_abs = op_elementwise<binary_op::MAX_ABS, 1.0, 1.0>;
using op_ew_min_abs = op_elementwise<binary_op::MIN_ABS, 1.0, 1.0>;
using op_ew_norm_2  = op_elementwise<binary_op::NORM_2, 1.0, 0.5>;
using op_ew_max     = op_elementwise<binary_op::MAX, 1.0, 1.0, false>;
using op_ew_min     = op_elementwise<binary_op::MIN, 1.0, 1.0, false>;

using op_tri_sum_sum     = op_elementwise_trinary<binary_op::SUM, binary_op::SUM, 1.0, 1.0, 0.5>;
using op_tri_prod_sum    = op_elementwise_trinary<binary_op::PROD, binary_op::SUM, 1.0, 1.0, 0.5>;
using op_tri_sum_prod    = op_elementwise_trinary<binary_op::SUM, binary_op::PROD, 1.0, 1.0, 1.0>;
using op_tri_prod_prod   = op_elementwise_trinary<binary_op::PROD, binary_op::PROD, 2.0, 2.0, 1.0>;
using op_tri_sum_abs_sum = op_elementwise_trinary<binary_op::SUM_ABS, binary_op::SUM, 1.0, 1.0, 0.5>;
using op_tri_norm_2_sum  = op_elementwise_trinary<binary_op::NORM_2, binary_op::SUM, 1.0, 1.0, 0.5>;
using op_tri_sum_norm_2  = op_elementwise_trinary<binary_op::SUM, binary_op::NORM_2, 1.0, 1.0, 0.5>;
using op_tri_max_sum     = op_elementwise_trinary<binary_op::MAX, binary_op::SUM, 1.0, 1.0, 0.5, false>;

using namespace nda_bench;

template <std::int32_t Rank>
using log_input = array_input<Rank, 'A', nda::C_layout, log_band>;
template <std::int32_t Rank>
using product_input = array_input<Rank, 'A', nda::C_layout, product_band>;

NDA_BENCHMARK(op_scale_identity, "scale_identity", types_double, array_input<2>)
NDA_BENCHMARK(op_scale_conj, "scale_conj", types_double, array_input<2>)
NDA_BENCHMARK(op_scale_abs, "scale_abs", types_double, array_input<2>)
NDA_BENCHMARK(op_scale_sqrt, "scale_sqrt", types_double, array_input<2>)
NDA_BENCHMARK(op_scale_exp, "scale_exp", types_double, array_input<2>)
NDA_BENCHMARK(op_scale_log, "scale_log", types_double, log_input<2>)
NDA_BENCHMARK(op_scale_rcp, "scale_rcp", types_double, array_input<2>)

NDA_BENCHMARK(op_ew_sum, "elementwise_sum", types_double, array_input<2>, array_input<2>)
NDA_BENCHMARK(op_ew_prod, "elementwise_prod", types_double, product_input<2>, array_input<2>)
NDA_BENCHMARK(op_ew_sum_abs, "elementwise_sum_abs", types_double, array_input<2>, array_input<2>)
NDA_BENCHMARK(op_ew_max_abs, "elementwise_max_abs", types_double, array_input<2>, array_input<2>)
NDA_BENCHMARK(op_ew_min_abs, "elementwise_min_abs", types_double, array_input<2>, array_input<2>)
NDA_BENCHMARK(op_ew_norm_2, "elementwise_norm_2", types_double, array_input<2>, array_input<2>)
NDA_BENCHMARK(op_ew_max, "elementwise_max", types<double>, array_input<2>, array_input<2>)
NDA_BENCHMARK(op_ew_min, "elementwise_min", types<double>, array_input<2>, array_input<2>)

NDA_BENCHMARK(op_tri_sum_sum, "elementwise_trinary_sum", types_double, array_input<2>, array_input<2>, array_input<2>)
NDA_BENCHMARK(op_tri_prod_sum, "elementwise_trinary_prod_sum", types_double, array_input<2>, array_input<2>, array_input<2>)
NDA_BENCHMARK(op_tri_sum_prod, "elementwise_trinary_sum_prod", types_double, product_input<2>, product_input<2>, array_input<2>)
NDA_BENCHMARK(op_tri_prod_prod, "elementwise_trinary_prod_prod", types_double, product_input<2>, product_input<2>, array_input<2>)
NDA_BENCHMARK(op_tri_sum_abs_sum, "elementwise_trinary_sum_abs_sum", types_double, array_input<2>, array_input<2>, array_input<2>)
NDA_BENCHMARK(op_tri_norm_2_sum, "elementwise_trinary_norm_2_sum", types_double, array_input<2>, array_input<2>, array_input<2>)
NDA_BENCHMARK(op_tri_sum_norm_2, "elementwise_trinary_sum_norm_2", types_double, array_input<2>, array_input<2>, array_input<2>)
NDA_BENCHMARK(op_tri_max_sum, "elementwise_trinary_max_sum", types<double>, array_input<2>, array_input<2>, array_input<2>)
