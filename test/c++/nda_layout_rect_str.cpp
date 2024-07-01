// Copyright (c) 2018 Commissariat à l'énergie atomique et aux énergies alternatives (CEA)
// Copyright (c) 2018 Centre national de la recherche scientifique (CNRS)
// Copyright (c) 2018-2023 Simons Foundation
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
// Authors: Thomas Hahn

#include "./test_common.hpp"

#include <nda/declarations.hpp>
#include <nda/layout/idx_map.hpp>
#include <nda/layout/permutation.hpp>
#include <nda/layout/rect_str.hpp>
#include <nda/traits.hpp>

#include <array>
#include <string>

using namespace nda;

// Make a std::array with long values.
template <typename... Is>
std::array<long, sizeof...(Is)> make_array(Is... is) {
  return {is...};
}

TEST(NDA, RectStrFromExistingIdxMap) {
  // create a rect_str from an existing idx_map
  idx_map<3, 0, C_stride_order<3>, layout_prop_e::contiguous> idxm{{2, 3, 4}};
  auto sidxm = rect_str(idxm);

  // check the basics
  EXPECT_EQ(sidxm.rank(), 3);
  EXPECT_EQ(sidxm.size(), 24);
  EXPECT_EQ(sidxm.ce_size(), 0);
  EXPECT_EQ(sidxm.lengths(), (make_array(2, 3, 4)));
  EXPECT_EQ(sidxm.strides(), (make_array(12, 4, 1)));
  EXPECT_EQ(sidxm.stride_order, (std::array<int, 3>{0, 1, 2}));
  EXPECT_EQ(sidxm.min_stride(), 1);
  EXPECT_TRUE(sidxm.is_contiguous());
  EXPECT_TRUE(sidxm.is_strided_1d());
  EXPECT_TRUE(sidxm.is_stride_order_C());
  EXPECT_TRUE(sidxm.is_stride_order_valid());

  // check the string indices
  auto idxs = sidxm.get_string_indices();
  for (int i = 0; i < sidxm.rank(); ++i) {
    for (int j = 0; j < sidxm.lengths()[i]; ++j) { EXPECT_EQ(idxs(i)(j), std::to_string(j)); }
  }

  // check the function call operator
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 4; ++k) {
        EXPECT_EQ(idxm(i, j, k), sidxm(std::to_string(i), std::to_string(j), std::to_string(k)));
        EXPECT_EQ(idxm(i, j, k), sidxm(i, std::to_string(j), k));
      }
    }
  }
}

TEST(NDA, RectStrFromGivenStringIndices) {
  // create a rect_str from string indices
  nda::array<nda::array<std::string, 1>, 1> idxs{{"a", "b"}, {"A", "B", "C"}, {"w", "x", "y", "z"}};
  rect_str<3, 0, Fortran_stride_order<3>, layout_prop_e::contiguous> sidxm(idxs);

  // check the basics
  EXPECT_EQ(sidxm.rank(), 3);
  EXPECT_EQ(sidxm.size(), 24);
  EXPECT_EQ(sidxm.ce_size(), 0);
  EXPECT_EQ(sidxm.lengths(), (make_array(2, 3, 4)));
  EXPECT_EQ(sidxm.strides(), (make_array(1, 2, 6)));
  EXPECT_EQ(sidxm.stride_order, (std::array<int, 3>{2, 1, 0}));
  EXPECT_EQ(sidxm.min_stride(), 1);
  EXPECT_TRUE(sidxm.is_contiguous());
  EXPECT_TRUE(sidxm.is_strided_1d());
  EXPECT_TRUE(sidxm.is_stride_order_Fortran());
  EXPECT_TRUE(sidxm.is_stride_order_valid());

  // check the function call operator
  for (int k = 0; k < 4; ++k) {
    for (int j = 0; j < 3; ++j) {
      for (int i = 0; i < 2; ++i) { EXPECT_EQ(sidxm(i, j, k), sidxm(idxs(0)(i), idxs(1)(j), idxs(2)(k))); }
    }
  }
}

TEST(NDA, RectStrFromDynamicExtentsOnly) {
  // create a rect_str from its dynamic extents only
  rect_str<2, encode(std::array{0, 10}), C_stride_order<2>, layout_prop_e::contiguous> sidxm(std::array{10l});

  // check the basics
  EXPECT_EQ(sidxm.rank(), 2);
  EXPECT_EQ(sidxm.size(), 100);
  EXPECT_EQ(sidxm.ce_size(), 0);
  EXPECT_EQ(sidxm.lengths(), (make_array(10, 10)));
  EXPECT_EQ(sidxm.strides(), (make_array(10, 1)));
  EXPECT_EQ(sidxm.stride_order, (std::array<int, 2>{0, 1}));
  EXPECT_EQ(sidxm.min_stride(), 1);
  EXPECT_TRUE(sidxm.is_contiguous());
  EXPECT_TRUE(sidxm.is_strided_1d());
  EXPECT_TRUE(sidxm.is_stride_order_C());
  EXPECT_TRUE(sidxm.is_stride_order_valid());

  // check the function call operator
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 3; ++j) { EXPECT_EQ(sidxm(i, j), sidxm(std::to_string(i), std::to_string(j))); }
  }
}

TEST(NDA, RectStrSlicesFrom3DMap) {
  // create a rect_str from string indices
  nda::array<nda::array<std::string, 1>, 1> idxs{{"a", "b"}, {"a", "b", "c"}, {"a", "b", "c", "d"}};
  rect_str<3, 0, C_stride_order<3>, layout_prop_e::contiguous> sidxm1(idxs);

  // take a trivial full slice (ellipsis are not working)
  // auto [offset2, r2] = r1.slice(ellipsis{});
  auto [offset2, sidxm2] = sidxm1.slice(range::all, range::all, range::all);
  EXPECT_EQ(offset2, 0);
  EXPECT_EQ(sidxm1, sidxm2);

  // take a strided 3D slice
  nda::array<nda::array<std::string, 1>, 1> idxs3D{{"a", "b"}, {"a", "c"}, {"b", "d"}};
  auto [offset3, sidxm3] = sidxm1.slice(range::all, range(0, 3, 2), range(1, 4, 2));
  EXPECT_EQ(offset3, 1);
  EXPECT_EQ(sidxm3.get_string_indices(), idxs3D);
  EXPECT_EQ(sidxm3.rank(), 3);
  EXPECT_EQ(sidxm3.lengths(), make_array(2, 2, 2));
  EXPECT_EQ(sidxm3.strides(), make_array(12, 8, 2));
  EXPECT_EQ(sidxm3.min_stride(), 2);
  EXPECT_EQ(sidxm3.layout_prop, layout_prop_e::none);

  // take a strided 2D slice
  nda::array<nda::array<std::string, 1>, 1> idxs2D{{"a", "c"}, {"b", "d"}};
  auto [offset4, sidxm4] = sidxm1.slice("a", range(0, 3, 2), range(1, 4, 2));
  EXPECT_EQ(offset4, 1);
  EXPECT_EQ(sidxm4.get_string_indices(), idxs2D);
  EXPECT_EQ(sidxm4.rank(), 2);
  EXPECT_EQ(sidxm4.lengths(), make_array(2, 2));
  EXPECT_EQ(sidxm4.strides(), make_array(8, 2));
  EXPECT_EQ(sidxm4.min_stride(), 2);
  EXPECT_EQ(sidxm4.layout_prop, layout_prop_e::none);
}
