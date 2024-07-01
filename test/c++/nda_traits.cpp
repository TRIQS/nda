// Copyright (c) 2023 Simons Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Authors: Nils Wentzell

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
