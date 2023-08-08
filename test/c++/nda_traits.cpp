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
#include <nda/traits.hpp>

// ==============================================================

// just compile time

TEST(NDA, Traits) { // NOLINT

  static_assert(nda::is_instantiation_of_v<std::vector, std::vector<double>>, "INTERNAL");

  static_assert(nda::is_any_of<int, int, double>, "INTERNAL");
  static_assert(nda::is_any_of<double, int, double>, "INTERNAL");

  static_assert(nda::is_complex_v<std::complex<float>>, "INTERNAL");
  static_assert(nda::is_complex_v<std::complex<double>>, "INTERNAL");
  static_assert(not nda::is_complex_v<double>, "INTERNAL");

  static_assert(not nda::is_scalar_v<std::string>, "INTERNAL");
  static_assert(nda::is_scalar_v<int>, "INTERNAL");
  static_assert(nda::is_scalar_v<double>, "INTERNAL");
  static_assert(nda::is_scalar_v<std::complex<double>>, "INTERNAL");

  static_assert(nda::get_rank<std::vector<double>> == 1, "INTERNAL");
  static_assert(nda::get_rank<nda::vector<double>> == 1, "INTERNAL");
  static_assert(nda::get_rank<nda::matrix<double>> == 2, "INTERNAL");
  static_assert(nda::get_rank<nda::array<double, 4>> == 4, "INTERNAL");

  static_assert(not nda::is_view_v<nda::array<double, 4>>, "INTERNAL");
  static_assert(nda::is_view_v<nda::array_view<double, 4>>, "INTERNAL");

  static_assert(std::is_same_v<nda::get_value_t<nda::vector<double>>, double>, "INTERNAL");
  static_assert(std::is_same_v<nda::get_value_t<double>, double>, "INTERNAL");
}
