// Copyright (c) 2024 Simons Foundation
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

#include <nda/layout/for_each.hpp>
#include <nda/layout/permutation.hpp>
#include <nda/stdutil/array.hpp>

#include <array>

TEST(NDA, ForEachIndexFromStrideOrder) {
  constexpr auto order_arr  = std::array{1, 2, 0};
  constexpr auto order_code = nda::encode(order_arr);
  static_assert(nda::detail::index_from_stride_order<3>(order_code, 0) == 1);
  static_assert(nda::detail::index_from_stride_order<3>(order_code, 1) == 2);
  static_assert(nda::detail::index_from_stride_order<3>(order_code, 2) == 0);
  EXPECT_EQ(nda::detail::index_from_stride_order<3>(order_code, 0), 1);
  EXPECT_EQ(nda::detail::index_from_stride_order<3>(order_code, 1), 2);
  EXPECT_EQ(nda::detail::index_from_stride_order<3>(order_code, 2), 0);
}

TEST(NDA, ForEachGetExtent) {
  constexpr auto shape_arr  = std::array{2, 3, 4};
  constexpr auto shape_code = nda::encode(shape_arr);
  constexpr auto zeros      = nda::stdutil::make_initialized_array<std::size(shape_arr)>(0l);
  static_assert(nda::detail::get_extent<0, 3, shape_code>(zeros) == 2);
  static_assert(nda::detail::get_extent<1, 3, shape_code>(zeros) == 3);
  static_assert(nda::detail::get_extent<2, 3, shape_code>(zeros) == 4);

  auto shape_dyn = shape_arr;
  auto e1        = nda::detail::get_extent<0, 3, 0>(shape_dyn);
  auto e2        = nda::detail::get_extent<1, 3, 0>(shape_dyn);
  auto e3        = nda::detail::get_extent<2, 3, 0>(shape_dyn);
  EXPECT_EQ(e1, 2);
  EXPECT_EQ(e2, 3);
  EXPECT_EQ(e3, 4);
}

TEST(NDA, ForeachStaticAndDynamic) {
  constexpr long n_i       = 2;
  constexpr long n_j       = 3;
  constexpr long n_k       = 4;
  constexpr auto shape_arr = std::array{n_i, n_j, n_k};

  // dynamic extents with C-order
  int count   = 0;
  auto flat_c = [](long i, long j, long k) { return i * n_j * n_k + j * n_k + k; };
  nda::for_each(shape_arr, [&count, flat_c](long i, long j, long k) {
    EXPECT_EQ(flat_c(i, j, k), count);
    ++count;
  });
  EXPECT_EQ(count, n_i * n_j * n_k);

  // static extents with C-order
  count                     = 0;
  constexpr auto shape_code = nda::encode(nda::stdutil::make_std_array<int>(shape_arr));
  nda::for_each_static<shape_code, 0>(shape_arr, [&count, flat_c](long i, long j, long k) {
    EXPECT_EQ(flat_c(i, j, k), count);
    ++count;
  });
  EXPECT_EQ(count, n_i * n_j * n_k);

  // static extents with Fortran-order
  count                       = 0;
  auto flat_f                 = [](long i, long j, long k) { return k * n_i * n_j + j * n_i + i; };
  constexpr auto f_order_arr  = std::array{2, 1, 0};
  constexpr auto f_order_code = nda::encode(f_order_arr);
  nda::for_each_static<shape_code, f_order_code>(shape_arr, [&count, flat_f](long i, long j, long k) {
    EXPECT_EQ(flat_f(i, j, k), count);
    ++count;
  });
  EXPECT_EQ(count, n_i * n_j * n_k);

  // dynamic extents with custom-order
  count                            = 0;
  auto flat_custom                 = [](long i, long j, long k) { return j * n_i * n_k + k * n_i + i; };
  constexpr auto custom_order_arr  = std::array{1, 2, 0};
  constexpr auto custom_order_code = nda::encode(custom_order_arr);
  nda::for_each_static<0, custom_order_code>(shape_arr, [&count, flat_custom](long i, long j, long k) {
    EXPECT_EQ(flat_custom(i, j, k), count);
    ++count;
  });
  EXPECT_EQ(count, n_i * n_j * n_k);
}

TEST(NDA, ForEachMutable) {
  std::array<long, 1> shape{10};
  std::vector<int> data(10, 0);
  nda::for_each(shape, [&, i = 0](long idx) mutable { data[idx] = i++; });
  for (int i = 0; i < 10; ++i) EXPECT_EQ(data[i], i);
}
