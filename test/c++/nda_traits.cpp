// Copyright (c) 2023--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#include "./test_common.hpp"

#include <nda/nda.hpp>
#include <nda/traits.hpp>

#include <complex>
#include <vector>

// Type convertible to std::complex<double>.
struct cplx_convertible {
  double x{1.0};
  operator std::complex<double>() const { return {x, 0.0}; }
};

// Type not convertible to std::complex<double>
struct not_cplx_convertible {
  double x{1.0};
};

TEST(NDA, TraitsGeneral) {
  static_assert(nda::is_instantiation_of_v<std::vector, std::vector<double>>);
  static_assert(nda::is_instantiation_of_v<std::complex, std::complex<double>>);
  static_assert(not nda::is_instantiation_of_v<std::complex, std::vector<float>>);

  static_assert(nda::is_any_of<int, int, double>);
  static_assert(not nda::is_any_of<float, int, double>);
  static_assert(not nda::is_any_of<int &, int, double>);

  static_assert(nda::always_true<int>);

  static_assert(nda::is_complex_v<std::complex<float>>);
  static_assert(nda::is_complex_v<std::complex<double>>);
  static_assert(not nda::is_complex_v<double>);

  static_assert(nda::is_scalar_v<int>);
  static_assert(nda::is_scalar_v<double &>);
  static_assert(nda::is_scalar_v<std::complex<long double> const &>);
  static_assert(not nda::is_scalar_v<std::vector<int>>);

  static_assert(nda::is_scalar_or_convertible_v<cplx_convertible>);
  static_assert(not nda::is_scalar_or_convertible_v<not_cplx_convertible>);

  static_assert(nda::is_double_or_complex_v<double const>);
  static_assert(not nda::is_double_or_complex_v<float>);
}

TEST(NDA, TraitsNDASpecific) {
  static_assert(nda::is_scalar_for_v<int, nda::vector<double>>);
  static_assert(not nda::is_scalar_for_v<std::complex<double>, nda::vector<cplx_convertible>>);
  static_assert(nda::is_scalar_for_v<cplx_convertible, nda::vector<std::complex<double>>>);
  static_assert(nda::is_scalar_for_v<not_cplx_convertible, nda::vector<not_cplx_convertible>>);
  static_assert(nda::is_scalar_for_v<int, nda::matrix<std::complex<double>>>);

  static_assert(nda::get_algebra<int> == 'N');
  static_assert(nda::get_algebra<nda::vector<double>> == 'V');
  static_assert(nda::get_algebra<nda::matrix<float> &> == 'M');
  static_assert(nda::get_algebra<nda::array<std::complex<double>, 4> const &> == 'A');

  static_assert(nda::get_rank<std::vector<double>> == 1);
  static_assert(nda::get_rank<nda::vector<double>> == 1);
  static_assert(nda::get_rank<nda::matrix<double>> == 2);
  static_assert(nda::get_rank<nda::array<double, 4>> == 4);

  static_assert(nda::is_regular_v<nda::vector<double>>);
  static_assert(nda::is_regular_v<nda::matrix<double>>);
  static_assert(nda::is_regular_v<nda::array<double, 4>>);
  static_assert(not nda::is_regular_v<nda::array_view<double, 4>>);

  static_assert(not nda::is_view_v<nda::vector<double>>);
  static_assert(not nda::is_view_v<nda::matrix<double>>);
  static_assert(not nda::is_view_v<nda::array<double, 4>>);
  static_assert(nda::is_view_v<nda::array_view<double, 4>>);

  static_assert(nda::is_matrix_or_view_v<nda::matrix<float>>);
  static_assert(not nda::is_matrix_or_view_v<nda::vector<float>>);

  EXPECT_EQ(nda::get_first_element(5), 5);
  EXPECT_EQ(nda::get_first_element(nda::vector<int>{5, 6, 7}), 5);
  EXPECT_EQ(nda::get_first_element(nda::array<int, 2>{{5, 6, 7}, {8, 9, 10}}), 5);

  static_assert(std::is_same_v<nda::get_value_t<int>, int>);
  static_assert(std::is_same_v<nda::get_value_t<nda::vector<double>>, double>);

  static_assert(nda::have_same_value_type_v<int, nda::vector<int>, nda::array_view<int, 2>>);
  static_assert(not nda::have_same_value_type_v<int, nda::vector<double>, nda::array_view<int, 2>>);

  static_assert(nda::have_same_value_type_v<int, nda::vector<int>, nda::array_view<int, 2>>);
  static_assert(not nda::have_same_value_type_v<int, nda::vector<double>, nda::array_view<int, 2>>);

  static_assert(nda::have_same_rank_v<nda::array<int, 2>, nda::array<double, 2>, nda::matrix<int>>);
  static_assert(not nda::have_same_rank_v<nda::array<int, 2>, nda::array<double, 2>, nda::vector<int>>);

  static_assert(nda::layout_property_compatible(nda::layout_prop_e::contiguous, nda::layout_prop_e::none));
  static_assert(nda::layout_property_compatible(nda::layout_prop_e::contiguous, nda::layout_prop_e::strided_1d));
  static_assert(nda::layout_property_compatible(nda::layout_prop_e::contiguous, nda::layout_prop_e::smallest_stride_is_one));
  static_assert(nda::layout_property_compatible(nda::layout_prop_e::contiguous, nda::layout_prop_e::contiguous));
  static_assert(nda::layout_property_compatible(nda::layout_prop_e::strided_1d, nda::layout_prop_e::strided_1d));
  static_assert(nda::layout_property_compatible(nda::layout_prop_e::strided_1d, nda::layout_prop_e::none));
  static_assert(not nda::layout_property_compatible(nda::layout_prop_e::strided_1d, nda::layout_prop_e::smallest_stride_is_one));
  static_assert(not nda::layout_property_compatible(nda::layout_prop_e::strided_1d, nda::layout_prop_e::contiguous));
  static_assert(nda::layout_property_compatible(nda::layout_prop_e::smallest_stride_is_one, nda::layout_prop_e::smallest_stride_is_one));
  static_assert(nda::layout_property_compatible(nda::layout_prop_e::smallest_stride_is_one, nda::layout_prop_e::none));
  static_assert(not nda::layout_property_compatible(nda::layout_prop_e::smallest_stride_is_one, nda::layout_prop_e::strided_1d));
  static_assert(not nda::layout_property_compatible(nda::layout_prop_e::smallest_stride_is_one, nda::layout_prop_e::contiguous));
  static_assert(nda::layout_property_compatible(nda::layout_prop_e::none, nda::layout_prop_e::none));
  static_assert(not nda::layout_property_compatible(nda::layout_prop_e::none, nda::layout_prop_e::strided_1d));
  static_assert(not nda::layout_property_compatible(nda::layout_prop_e::none, nda::layout_prop_e::smallest_stride_is_one));
  static_assert(not nda::layout_property_compatible(nda::layout_prop_e::none, nda::layout_prop_e::contiguous));

  static_assert((nda::layout_prop_e::contiguous & nda::layout_prop_e::none) == nda::layout_prop_e::none);
  static_assert((nda::layout_prop_e::contiguous & nda::layout_prop_e::strided_1d) == nda::layout_prop_e::strided_1d);
  static_assert((nda::layout_prop_e::contiguous & nda::layout_prop_e::smallest_stride_is_one) == nda::layout_prop_e::smallest_stride_is_one);

  static_assert((nda::layout_prop_e::strided_1d | nda::layout_prop_e::smallest_stride_is_one) == nda::layout_prop_e::contiguous);
  static_assert((nda::layout_prop_e::contiguous | nda::layout_prop_e::none) == nda::layout_prop_e::contiguous);

  static_assert(nda::has_contiguous(nda::layout_prop_e::contiguous));
  static_assert(not nda::has_contiguous(nda::layout_prop_e::none));
  static_assert(nda::has_strided_1d(nda::layout_prop_e::strided_1d));
  static_assert(nda::has_strided_1d(nda::layout_prop_e::contiguous));
  static_assert(not nda::has_strided_1d(nda::layout_prop_e::smallest_stride_is_one));
  static_assert(nda::has_smallest_stride_is_one(nda::layout_prop_e::smallest_stride_is_one));
  static_assert(nda::has_smallest_stride_is_one(nda::layout_prop_e::contiguous));
  static_assert(not nda::has_smallest_stride_is_one(nda::layout_prop_e::strided_1d));

  constexpr nda::layout_info_t cinfo{2, nda::layout_prop_e::contiguous};
  constexpr nda::layout_info_t sinfo{2, nda::layout_prop_e::strided_1d};
  constexpr nda::layout_info_t sinfo_2{1, nda::layout_prop_e::strided_1d};
  static_assert((cinfo & sinfo).prop == nda::layout_prop_e::strided_1d);
  static_assert((cinfo & sinfo).stride_order == 2);
  static_assert((cinfo & sinfo_2).prop == nda::layout_prop_e::none);
  static_assert((cinfo & sinfo_2).stride_order == static_cast<uint64_t>(-1));
}

template <typename T, typename Layout1, typename Layout2>
void check_all_simd_traits() {
  using namespace nda;
  constexpr int Rank = 2;
  std::array<long, Rank> shape;
  shape.fill(64);

  using A_t = array<T, Rank, Layout1>;
  using B_t = array<T, Rank, Layout2>;

  A_t A(shape);
  B_t B(shape);

  static_assert(is_simd_enabled_v<A_t>);
  static_assert(is_simd_enabled_v<B_t>);

  constexpr bool layouts_match = std::is_same_v<Layout1, Layout2>;

  auto add_expr = A + B;
  auto sub_expr = A - B;
  auto mul_expr = A * B;
  auto div_expr = A / B;

  static_assert(is_simd_enabled_v<decltype(add_expr)> == layouts_match);
  static_assert(is_simd_enabled_v<decltype(sub_expr)> == layouts_match);
  static_assert(is_simd_enabled_v<decltype(mul_expr)> == layouts_match);
  static_assert(is_simd_enabled_v<decltype(div_expr)> == layouts_match);

  static_assert(is_simd_enabled_v<decltype(add_expr * sub_expr)> == layouts_match);
  static_assert(is_simd_enabled_v<decltype((sub_expr / div_expr) + mul_expr)> == layouts_match);
  static_assert(is_simd_enabled_v<decltype(mul_expr + mul_expr * add_expr)> == layouts_match);
  static_assert(is_simd_enabled_v<decltype(-(-div_expr + T{3}) * T{2})> == layouts_match);

  auto s_mul      = A * T{2};
  auto s_mul_left = T{2} * A;
  static_assert(is_simd_enabled_v<decltype(s_mul)>);
  static_assert(is_simd_enabled_v<decltype(s_mul_left)>);

  auto neg_expr = -A;
  static_assert(is_simd_enabled_v<decltype(neg_expr)>);

  array<int16_t, Rank, Layout1> wrong(shape);
  auto mixed_expr = A + wrong;
  static_assert(!is_simd_enabled_v<decltype(mixed_expr)>);
  static_assert(!is_simd_enabled_v<decltype(mixed_expr * add_expr)>);
  static_assert(!is_simd_enabled_v<decltype(add_expr + mixed_expr - wrong)>);

  auto V = A(range(0, 16), 0);
  static_assert(!is_simd_enabled_v<decltype(V)>);
  array<T, 1> slice_size(std::array{16});
  auto sliced_expr = V + slice_size;
  static_assert(!is_simd_enabled_v<decltype(sliced_expr)>);

  // ─── Map functors ──────────────────────────────────────
  struct no_load_f {
    T operator()(T x, T y) const { return x + y; }
  };
  struct mock_only_f : simd::mock_simd<mock_only_f, T> {
    T operator()(T x, T y) const { return x + y; }
  };
  struct native_simd_f {
    T operator()(T x, T y) const { return x + y; }
    native_simd<T> load(native_simd<T> x, native_simd<T> y) const { return x + y; }
  };
  struct wrong_sig_f {
    T operator()(T x, T y) const { return x + y; }
    T load(T x, T y) const { return x + y; }
  };
  struct bad_load_return_simd {
    T operator()(T x, T y) const { return x + y; }
    native_simd<T> load(T x, T y) const { return native_simd<T>{x + y}; }
  };
  struct bad_load_return_scalar {
    T operator()(T x, T y) const { return x + y; }
    T load(native_simd<T> x, native_simd<T> y) const { return (x + y).get(0); }
  };

  auto m1 = map(no_load_f{})(A, B);
  static_assert(!is_simd_enabled_v<decltype(m1)>);

  auto m2 = map(mock_only_f{})(A, B);
  static_assert(is_simd_enabled_v<decltype(m2)> == layouts_match);

  auto m3 = map(native_simd_f{})(A, B);
  static_assert(is_simd_enabled_v<decltype(m3)> == layouts_match);

  auto m4 = map(wrong_sig_f{})(A, B);
  static_assert(!is_simd_enabled_v<decltype(m4)>);

  auto m5 = map(bad_load_return_simd{})(A, B);
  static_assert(!is_simd_enabled_v<decltype(m5)>);

  auto m6 = map(bad_load_return_scalar{})(A, B);
  static_assert(!is_simd_enabled_v<decltype(m6)>);

  auto deep_map_expr = map(native_simd_f{})(A, B) + A * B - T{3} + add_expr;
  static_assert(is_simd_enabled_v<decltype(deep_map_expr)> == layouts_match);

  auto broken_expr = map(native_simd_f{})(A, B) + map(no_load_f{})(A, B) + mul_expr;
  static_assert(!is_simd_enabled_v<decltype(broken_expr)>);

  auto mock_combined = map(mock_only_f{})(A, B) * T{2} - T{1};
  static_assert(is_simd_enabled_v<decltype(mock_combined)> == layouts_match);

  auto map_then_slice = map(native_simd_f{})(A, B)(range(0, 16), 0);
  static_assert(!is_simd_enabled_v<decltype(map_then_slice)>);

  // ─── Map applied to expressions ────────────────────────────────
  auto map1 = map(no_load_f{})(add_expr, mul_expr);
  static_assert(!is_simd_enabled_v<decltype(map1)>);

  auto map2 = map(wrong_sig_f{})(div_expr, B);
  static_assert(!is_simd_enabled_v<decltype(map2)>);

  auto map3 = map(mock_only_f{})(A, sub_expr);
  static_assert(is_simd_enabled_v<decltype(map3)> == layouts_match);

  auto map4 = map(native_simd_f{})(add_expr, B);
  static_assert(is_simd_enabled_v<decltype(map4)> == layouts_match);

  auto m1_plus_mul = map(no_load_f{})(A, sub_expr) + mul_expr;
  static_assert(!is_simd_enabled_v<decltype(m1_plus_mul)>);

  auto m2_times_div = map(wrong_sig_f{})(add_expr, A) * div_expr;
  static_assert(!is_simd_enabled_v<decltype(m2_times_div)>);

  auto m3_plus_mul = map(mock_only_f{})(mul_expr, B) + sub_expr;
  static_assert(is_simd_enabled_v<decltype(m3_plus_mul)> == layouts_match);

  auto m4_plus_sub = map(native_simd_f{})(A, div_expr) + sub_expr;
  static_assert(is_simd_enabled_v<decltype(m4_plus_sub)> == layouts_match);

  auto m5_plus_add = map(bad_load_return_simd{})(add_expr, mul_expr) + A;
  static_assert(!is_simd_enabled_v<decltype(m5_plus_add)>);

  auto m6_plus_sub = map(bad_load_return_scalar{})(sub_expr, B) + div_expr;
  static_assert(!is_simd_enabled_v<decltype(m6_plus_sub)>);
}


TEST(NDA, SIMD_TRAITS) {

#define TEST_SIMD_TRAITS_FOR_LAYOUTS(type)                                                                                                           \
  check_all_simd_traits<type, nda::C_layout, nda::C_layout>();                                                                                       \
  check_all_simd_traits<type, nda::C_layout, nda::F_layout>();                                                                                       \
  check_all_simd_traits<type, nda::F_layout, nda::C_layout>();                                                                                       \
  check_all_simd_traits<type, nda::F_layout, nda::F_layout>();

  TEST_SIMD_TRAITS_FOR_LAYOUTS(int32_t)
  TEST_SIMD_TRAITS_FOR_LAYOUTS(int64_t)
  TEST_SIMD_TRAITS_FOR_LAYOUTS(uint32_t)
  TEST_SIMD_TRAITS_FOR_LAYOUTS(uint64_t)
  TEST_SIMD_TRAITS_FOR_LAYOUTS(float)
  TEST_SIMD_TRAITS_FOR_LAYOUTS(double)
  TEST_SIMD_TRAITS_FOR_LAYOUTS(std::complex<float>)
  TEST_SIMD_TRAITS_FOR_LAYOUTS(std::complex<double>)

#undef TEST_SIMD_TRAITS_FOR_LAYOUTS
}
