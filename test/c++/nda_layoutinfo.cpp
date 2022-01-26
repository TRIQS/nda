// Copyright (c) 2020-2021 Simons Foundation
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
// Authors: Olivier Parcollet, Nils Wentzell

#include "./test_common.hpp"
// -------------------------------------

TEST(BUG, n1) { //NOLINT

  using expr_t = nda::expr<'+', nda::basic_array<std::complex<double>, 2, nda::C_layout, 'M', nda::heap<>> &,
                           const nda::basic_array_view<const std::complex<double>, 2, nda::basic_layout<0, 16, nda::layout_prop_e::none>, 'A',
                                                       nda::default_accessor, nda::borrowed<>> &>;

  EXPECT_EQ((nda::layout_prop_e::none & nda::layout_prop_e::contiguous), nda::layout_prop_e::none);

  EXPECT_EQ(expr_t::l_is_scalar, false);
  EXPECT_EQ(expr_t::r_is_scalar, false);

  EXPECT_EQ(expr_t::compute_layout_info().prop, nda::layout_prop_e::none);

  EXPECT_EQ(nda::get_layout_info<expr_t::R_t>.prop, nda::layout_prop_e::none);
  EXPECT_EQ(nda::get_layout_info<expr_t::L_t>.prop, nda::layout_prop_e::contiguous);

  EXPECT_EQ(nda::get_layout_info<expr_t::R_t>.prop & nda::get_layout_info<expr_t::L_t>.prop, nda::layout_prop_e::none);

  //EXPECT_EQ(nda::get_layout_info<expr_t>.prop, nda::layout_prop_e::none);
  //EXPECT_EQ(nda::get_layout_info<expr_t>.prop, nda::layout_prop_e::contiguous);
}
