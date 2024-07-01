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
// Authors: Dominik Kiese, Olivier Parcollet, Nils Wentzell

#include "./test_common.hpp"

#include <nda/layout/idx_map.hpp>
#include <nda/layout/permutation.hpp>
#include <nda/traits.hpp>

#include <array>
#include <tuple>

using namespace nda;

// Make a std::array with long values.
template <typename... Is>
std::array<long, sizeof...(Is)> make_array(Is... is) {
  return {is...};
}

TEST(NDA, IdxMapContiguousCOrder) {
  // contiguous 2x3x4 index map in C-order
  idx_map<3, 0, C_stride_order<3>, layout_prop_e::contiguous> idxm{{2, 3, 4}};
  EXPECT_EQ(idxm.rank(), 3);
  EXPECT_EQ(idxm.size(), 24);
  EXPECT_EQ(idxm.ce_size(), 0);
  EXPECT_EQ(idxm.lengths(), make_array(2, 3, 4));
  EXPECT_EQ(idxm.strides(), make_array(12, 4, 1));
  EXPECT_EQ(idxm.stride_order, (std::array<int, 3>{0, 1, 2}));
  EXPECT_EQ(idxm.min_stride(), 1);
  EXPECT_TRUE(idxm.is_contiguous());
  EXPECT_TRUE(idxm.is_strided_1d());
  EXPECT_TRUE(idxm.is_stride_order_C());
  EXPECT_TRUE(idxm.is_stride_order_valid());
  long flat_idx = 0;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 4; ++k) {
        EXPECT_EQ(idxm(i, j, k), 12 * i + 4 * j + k);
        EXPECT_EQ(idxm.to_idx(flat_idx++), make_array(i, j, k));
      }
    }
  }
}

TEST(NDA, IdxMapContiguousFortranOrder) {
  // contiguous 2x3x4 index map in Fortran-order
  idx_map<3, encode(std::array{2, 3, 4}), Fortran_stride_order<3>, layout_prop_e::contiguous> idxm{};
  EXPECT_EQ(idxm.rank(), 3);
  EXPECT_EQ(idxm.size(), 24);
  EXPECT_EQ(idxm.ce_size(), 24);
  EXPECT_EQ(idxm.lengths(), make_array(2, 3, 4));
  EXPECT_EQ(idxm.strides(), make_array(1, 2, 6));
  EXPECT_EQ(idxm.stride_order, (std::array<int, 3>{2, 1, 0}));
  EXPECT_EQ(idxm.min_stride(), 1);
  EXPECT_TRUE(idxm.is_contiguous());
  EXPECT_TRUE(idxm.is_strided_1d());
  EXPECT_TRUE(idxm.is_stride_order_Fortran());
  EXPECT_TRUE(idxm.is_stride_order_valid());
  long flat_idx = 0;
  for (int k = 0; k < 4; ++k) {
    for (int j = 0; j < 3; ++j) {
      for (int i = 0; i < 2; ++i) {
        EXPECT_EQ(idxm(i, j, k), 6 * k + 2 * j + i);
        EXPECT_EQ(idxm.to_idx(flat_idx++), make_array(i, j, k));
      }
    }
  }
}

TEST(NDA, IdxMapContiguousArbitraryOrder) {
  // contiguous 2x3x4 index map in arbitrary order
  idx_map<3, encode(std::array{2, 0, 4}), encode(std::array{2, 0, 1}), layout_prop_e::contiguous> idxm{{2, 3, 4}};
  EXPECT_EQ(idxm.rank(), 3);
  EXPECT_EQ(idxm.size(), 24);
  EXPECT_EQ(idxm.ce_size(), 0);
  EXPECT_EQ(idxm.lengths(), make_array(2, 3, 4));
  EXPECT_EQ(idxm.strides(), make_array(3, 1, 6));
  EXPECT_EQ(idxm.stride_order, (std::array<int, 3>{2, 0, 1}));
  EXPECT_EQ(idxm.min_stride(), 1);
  EXPECT_TRUE(idxm.is_contiguous());
  EXPECT_TRUE(idxm.is_strided_1d());
  EXPECT_FALSE(idxm.is_stride_order_C());
  EXPECT_FALSE(idxm.is_stride_order_Fortran());
  EXPECT_TRUE(idxm.is_stride_order_valid());
  long flat_idx = 0;
  for (int k = 0; k < 4; ++k) {
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 3; ++j) {
        EXPECT_EQ(idxm(i, j, k), 6 * k + 3 * i + j);
        EXPECT_EQ(idxm.to_idx(flat_idx++), make_array(i, j, k));
      }
    }
  }
}

TEST(NDA, IdxMapStridedCOrder) {
  // strided 5x5 index map in C-order with underlying 10x10 contiguous index map: every 2nd element in each dimension
  idx_map<2, 0, C_stride_order<2>, layout_prop_e::none> idxm{{5, 5}, {10, 2}};
  EXPECT_EQ(idxm.rank(), 2);
  EXPECT_EQ(idxm.size(), 25);
  EXPECT_EQ(idxm.ce_size(), 0);
  EXPECT_EQ(idxm.lengths(), make_array(5, 5));
  EXPECT_EQ(idxm.strides(), make_array(10, 2));
  EXPECT_EQ(idxm.stride_order, (std::array<int, 2>{0, 1}));
  EXPECT_EQ(idxm.min_stride(), 2);
  EXPECT_FALSE(idxm.is_contiguous());
  EXPECT_TRUE(idxm.is_strided_1d());
  EXPECT_TRUE(idxm.is_stride_order_C());
  EXPECT_TRUE(idxm.is_stride_order_valid());
  long flat_idx = 0;
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      EXPECT_EQ(idxm(i, j), 10 * i + 2 * j);
      EXPECT_EQ(idxm.to_idx(flat_idx), make_array(i, j));
      flat_idx += 2;
    }
  }
}

TEST(NDA, IdxMapStridedFortranOrder) {
  // strided 10x5 index map in Fortran-order with underlying 10x10 contiguous index map: every 2nd column
  idx_map<2, 0, Fortran_stride_order<2>, layout_prop_e::none> idxm{{10, 5}, {1, 20}};
  EXPECT_EQ(idxm.rank(), 2);
  EXPECT_EQ(idxm.size(), 50);
  EXPECT_EQ(idxm.ce_size(), 0);
  EXPECT_EQ(idxm.lengths(), make_array(10, 5));
  EXPECT_EQ(idxm.strides(), make_array(1, 20));
  EXPECT_EQ(idxm.stride_order, (std::array<int, 2>{1, 0}));
  EXPECT_EQ(idxm.min_stride(), 1);
  EXPECT_FALSE(idxm.is_contiguous());
  EXPECT_FALSE(idxm.is_strided_1d());
  EXPECT_TRUE(idxm.is_stride_order_Fortran());
  EXPECT_TRUE(idxm.is_stride_order_valid());
  long flat_idx = 0;
  for (int j = 0; j < 5; ++j) {
    for (int i = 0; i < 10; ++i) {
      EXPECT_EQ(idxm(i, j), 1 * i + 20 * j);
      EXPECT_EQ(idxm.to_idx(flat_idx), make_array(i, j));
      flat_idx += 1;
      if (i == 9) flat_idx += 10;
    }
  }
}

TEST(NDA, IdxMapStridedArbitraryOrder) {
  // strided 10x5x4 idx_map in arbitrary order with underlying 10x10x10 contiguous idx_map:
  // every 1st element in 1st dimension, every 2nd element in 2nd dimension, every 3rd element in 3rd dimension
  idx_map<3, 0, encode(std::array{1, 0, 2}), layout_prop_e::none> idxm{{10, 5, 4}, {10, 200, 3}};
  EXPECT_EQ(idxm.rank(), 3);
  EXPECT_EQ(idxm.size(), 200);
  EXPECT_EQ(idxm.ce_size(), 0);
  EXPECT_EQ(idxm.lengths(), make_array(10, 5, 4));
  EXPECT_EQ(idxm.strides(), make_array(10, 200, 3));
  EXPECT_EQ(idxm.stride_order, (std::array<int, 3>{1, 0, 2}));
  EXPECT_EQ(idxm.min_stride(), 3);
  EXPECT_FALSE(idxm.is_contiguous());
  EXPECT_FALSE(idxm.is_strided_1d());
  EXPECT_FALSE(idxm.is_stride_order_C());
  EXPECT_FALSE(idxm.is_stride_order_Fortran());
  EXPECT_TRUE(idxm.is_stride_order_valid());
  long flat_idx = 0;
  for (int j = 0; j < 5; ++j) {
    for (int i = 0; i < 10; ++i) {
      for (int k = 0; k < 4; ++k) {
        EXPECT_EQ(idxm(i, j, k), 10 * i + 200 * j + 3 * k);
        EXPECT_EQ(idxm.to_idx(flat_idx), make_array(i, j, k));
        if (k == 3) {
          flat_idx += 1;
          if (i == 9) { flat_idx += 100; }
        } else {
          flat_idx += 3;
        }
      }
    }
  }
}

TEST(NDA, IdxMapTranspose) {
  // transpose a contiguous 2x3x4 idx_map in C-order with the permutation (1, 2, 0) - its inverse is (2, 0, 1)
  constexpr auto perm     = std::array{1, 2, 0};
  constexpr auto inv_perm = permutations::inverse(perm);
  idx_map<3, 0, C_stride_order<3>, layout_prop_e::contiguous> idxm1{{2, 3, 4}};
  auto idxm2 = idxm1.transpose<encode(perm)>();
  EXPECT_EQ(idxm2.rank(), 3);
  EXPECT_EQ(idxm2.size(), 24);
  EXPECT_EQ(idxm2.ce_size(), 0);
  EXPECT_EQ(idxm2.lengths(), make_array(4, 2, 3));
  EXPECT_EQ(idxm2.strides(), make_array(1, 12, 4));
  EXPECT_EQ(idxm2.stride_order, (std::array<int, 3>{1, 2, 0}));
  EXPECT_EQ(idxm2.min_stride(), 1);
  EXPECT_TRUE(idxm2.is_contiguous());
  EXPECT_TRUE(idxm2.is_strided_1d());
  EXPECT_FALSE(idxm2.is_stride_order_C());
  EXPECT_FALSE(idxm2.is_stride_order_Fortran());
  EXPECT_TRUE(idxm2.is_stride_order_valid());
  long flat_idx = 0;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 3; ++k) {
        EXPECT_EQ(idxm2(i, j, k), std::apply([&idxm1](auto... idx) { return idxm1(idx...); }, permutations::apply(perm, std::array{i, j, k})));
        EXPECT_EQ(idxm2.to_idx(flat_idx), permutations::apply(inv_perm, idxm1.to_idx(flat_idx)));
        flat_idx += 1;
      }
    }
  }
}

TEST(NDA, IdxMapSlicesFrom1DIdxMap) {
  // take slices of a 1D index map
  idx_map<1, 0, C_stride_order<1>, layout_prop_e::contiguous> idxm{{100}};

  // take a trivial full slice
  auto [offset1, slice1] = idxm.slice(range::all);
  EXPECT_EQ(offset1, 0);
  EXPECT_EQ(slice1, idxm);
  EXPECT_EQ(slice1.layout_prop, idxm.layout_prop);

  // take a 1D strided slice
  auto rg                = range(4, 100, 5);
  auto [offset2, slice2] = idxm.slice(rg);
  EXPECT_EQ(offset2, 4);
  EXPECT_EQ(slice2.rank(), 1);
  EXPECT_EQ(slice2.size(), rg.size());
  EXPECT_EQ(slice2.lengths(), make_array(rg.size()));
  EXPECT_EQ(slice2.strides(), make_array(rg.step()));
  EXPECT_EQ(slice2.min_stride(), rg.step());
  EXPECT_FALSE(slice2.is_contiguous());
  EXPECT_TRUE(slice2.is_strided_1d());
}

TEST(NDA, IdxMapSlicesFrom2DIdxMap) {
  // take slices of a 2D index map
  idx_map<2, 0, Fortran_stride_order<2>, layout_prop_e::contiguous> idxm{{100, 100}};

  // take a trivial full slice
  auto [offset1, slice1] = idxm.slice(ellipsis{});
  EXPECT_EQ(offset1, 0);
  EXPECT_EQ(slice1, idxm);
  EXPECT_EQ(slice1.layout_prop, idxm.layout_prop);

  // take a 2D slice with unnecessary ellipsis
  auto rg1               = range(4, 100, 5);
  auto rg2               = range(10, 100, 10);
  auto [offset2, slice2] = idxm.slice(rg1, ellipsis{}, rg2);
  EXPECT_EQ(offset2, 1004);
  EXPECT_EQ(slice2.rank(), 2);
  EXPECT_EQ(slice2.size(), rg1.size() * rg2.size());
  EXPECT_EQ(slice2.lengths(), make_array(rg1.size(), rg2.size()));
  EXPECT_EQ(slice2.strides(), make_array(rg1.step(), 100 * rg2.step()));
  EXPECT_EQ(slice2.min_stride(), rg1.step());
  EXPECT_EQ(slice2.layout_prop, layout_prop_e::none);
  EXPECT_TRUE(slice2.is_stride_order_Fortran());
  EXPECT_TRUE(slice2.is_stride_order_valid());

  // take a single column
  auto [offset3, slice3] = idxm.slice(range::all, 10);
  EXPECT_EQ(offset3, 1000);
  EXPECT_EQ(slice3.rank(), 1);
  EXPECT_EQ(slice3.size(), 100);
  EXPECT_EQ(slice3.lengths(), make_array(100));
  EXPECT_EQ(slice3.strides(), make_array(1));
  EXPECT_EQ(slice3.min_stride(), 1);
  EXPECT_EQ(slice3.layout_prop, layout_prop_e::contiguous);

  // take every 2nd element of a single row
  auto rg3               = range(0, 100, 2);
  auto [offset4, slice4] = idxm.slice(10, rg3);
  EXPECT_EQ(offset4, 10);
  EXPECT_EQ(slice4.rank(), 1);
  EXPECT_EQ(slice4.size(), rg3.size());
  EXPECT_EQ(slice4.lengths(), make_array(rg3.size()));
  EXPECT_EQ(slice4.strides(), make_array(200));
  EXPECT_EQ(slice4.min_stride(), 200);
  EXPECT_EQ(slice4.layout_prop, layout_prop_e::strided_1d);
}

TEST(NDA, IdxMapSlicesFrom3DIdxMap) {
  // take slices of a 3D index map
  idx_map<3, 0, encode(std::array{1, 2, 0}), layout_prop_e::contiguous> idxm{{100, 100, 100}};

  // take a trivial full slice
  auto [offset1, slice1] = idxm.slice(ellipsis{});
  EXPECT_EQ(offset1, 0);
  EXPECT_EQ(slice1, idxm);
  EXPECT_EQ(slice1.layout_prop, idxm.layout_prop);
  EXPECT_EQ(slice1.stride_order, idxm.stride_order);

  // take a strided 3D slice
  auto rg1               = range(4, 100, 5);
  auto rg2               = range(10, 100, 10);
  auto rg3               = range(0, 100, 2);
  auto [offset2, slice2] = idxm.slice(rg1, rg2, rg3);
  EXPECT_EQ(offset2, 100 * 100 * 10 + 4);
  EXPECT_EQ(slice2.rank(), 3);
  EXPECT_EQ(slice2.lengths(), make_array(rg1.size(), rg2.size(), rg3.size()));
  EXPECT_EQ(slice2.strides(), make_array(5, 100000, 200));
  EXPECT_EQ(slice2.min_stride(), 5);
  EXPECT_EQ(slice2.layout_prop, layout_prop_e::none);

  // take a 3D slice with ellipsis
  auto rg4               = range(4, 100, 5);
  auto [offset3, slice3] = idxm.slice(rg4, ellipsis{});
  EXPECT_EQ(offset3, 4);
  EXPECT_EQ(slice3.rank(), 3);
  EXPECT_EQ(slice3.lengths(), make_array(rg4.size(), 100, 100));
  EXPECT_EQ(slice3.strides(), make_array(5, 10000, 100));
  EXPECT_EQ(slice3.min_stride(), 5);
  EXPECT_EQ(slice3.layout_prop, layout_prop_e::none);

  // take a contiguous 2D slice
  auto [offset4, slice4] = idxm.slice(range::all, 10, range::all);
  EXPECT_EQ(offset4, 100000);
  EXPECT_EQ(slice4.rank(), 2);
  EXPECT_EQ(slice4.lengths(), make_array(100, 100));
  EXPECT_EQ(slice4.strides(), make_array(1, 100));
  EXPECT_EQ(slice4.stride_order, (std::array<int, 2>{1, 0}));
  EXPECT_EQ(slice4.min_stride(), 1);
  EXPECT_EQ(slice4.layout_prop, layout_prop_e::contiguous);

  // take a non-contiguous 2D slice
  auto [offset5, slice5] = idxm.slice(50, ellipsis{});
  EXPECT_EQ(offset5, 50);
  EXPECT_EQ(slice5.rank(), 2);
  EXPECT_EQ(slice5.lengths(), make_array(100, 100));
  EXPECT_EQ(slice5.strides(), make_array(10000, 100));
  EXPECT_EQ(slice5.stride_order, (std::array<int, 2>{0, 1}));
  EXPECT_EQ(slice5.min_stride(), 100);
  EXPECT_EQ(slice5.layout_prop, layout_prop_e::strided_1d);

  // take a contiguous 1D slice
  auto [offset6, slice6] = idxm.slice(range::all, 10, 10);
  EXPECT_EQ(offset6, 100 * 10 + 10000 * 10);
  EXPECT_EQ(slice6.rank(), 1);
  EXPECT_EQ(slice6.lengths(), make_array(100));
  EXPECT_EQ(slice6.strides(), make_array(1));
  EXPECT_EQ(slice6.min_stride(), 1);
  EXPECT_EQ(slice6.layout_prop, layout_prop_e::contiguous);

  // take a strided 1D slice
  auto rg5               = range(0, 100, 2);
  auto [offset7, slice7] = idxm.slice(10, 10, rg5);
  EXPECT_EQ(offset7, 100 * 100 * 10 + 10);
  EXPECT_EQ(slice7.rank(), 1);
  EXPECT_EQ(slice7.lengths(), make_array(rg5.size()));
  EXPECT_EQ(slice7.strides(), make_array(200));
  EXPECT_EQ(slice7.min_stride(), 200);
  EXPECT_EQ(slice7.layout_prop, layout_prop_e::strided_1d);
}

TEST(NDA, IdxMapOtherSlices) {
  // take a 1D slice of a 2D index map
  idx_map<2, 0, C_stride_order<2>, layout_prop_e::none> idxm1{{10, 10}};
  auto [offset1, slice1] = slice_static::slice_idx_map(idxm1, range(0, 2), 2);
  static_assert(decltype(slice1)::layout_prop == layout_prop_e::strided_1d);

  // take a 1D slice of a 3D index map
  idx_map<3, 0, C_stride_order<3>, layout_prop_e::none> idxm2{{1, 2, 3}};
  auto [offset2, slice2] = slice_static::slice_idx_map(idxm2, 0, range::all, 2);
  idx_map<1, 0, C_stride_order<1>, layout_prop_e::strided_1d> exp2{{2}, {3}};
  EXPECT_TRUE(slice2 == exp2);
  EXPECT_EQ(offset2, 2);

  // take a full slice of a 3D index map
  auto [offset3, slice3] = slice_static::slice_idx_map(idxm2, range::all, range::all, range::all);
  EXPECT_TRUE(slice3 == idxm2);
  EXPECT_EQ(offset3, 0);

  // take a 2D slice of a 3D index map
  auto [offset4, slice4] = slice_static::slice_idx_map(idxm2, 0, ellipsis{});
  idx_map<2, 0, C_stride_order<2>, layout_prop_e::none> exp4{{2, 3}, {3, 1}};
  EXPECT_TRUE(slice4 == exp4);
  EXPECT_EQ(offset4, 0);

  // take a full slice of a 3D index map with ellipsis
  auto [offset5, slice5] = slice_static::slice_idx_map(idxm2, ellipsis{});
  EXPECT_TRUE(slice5 == idxm2);
  EXPECT_EQ(offset5, 0);

  // take a 2D slice of a 5D index map
  idx_map<5, 0, C_stride_order<5>, layout_prop_e::none> idxm3{{1, 2, 3, 4, 5}};
  auto [offset6, slice6] = slice_static::slice_idx_map(idxm3, 0, ellipsis{}, 3, 2);
  idx_map<2, 0, C_stride_order<2>, layout_prop_e::none> exp6{{2, 3}, {60, 20}};
  EXPECT_TRUE(slice6 == exp6);
  EXPECT_EQ(offset6, idxm3(0, 0, 0, 3, 2));
}
