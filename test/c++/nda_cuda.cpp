// Copyright (c) 2022-2023 Simons Foundation
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
// Authors: Miguel Morales, Nils Wentzell

#include "./test_common.hpp"

#include <nda/gtest_tools.hpp>
#include <nda/nda.hpp>

using value_t   = double;
constexpr int N = 4;

template <size_t Rank>
using cuarray_t = nda::cuarray<value_t, Rank>;

template <size_t Rank>
using cuarray_vt = nda::cuarray_view<value_t, Rank>;

template <size_t Rank>
using cuarray_cvt = nda::cuarray_const_view<value_t, Rank>;

template <size_t Rank>
using array_t = nda::array<value_t, Rank>;

template <size_t Rank>
using array_vt = nda::array_view<value_t, Rank>;

template <size_t Rank>
using array_cvt = nda::array_view<const value_t, Rank>;

template <size_t Rank>
using unmarray_t = nda::basic_array<value_t, Rank, nda::C_layout, 'A', nda::heap<nda::mem::Unified>>;

template <size_t Rank>
using unmarray_vt = nda::basic_array_view<value_t, Rank, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::Unified>>;

template <size_t Rank>
using unmarray_cvt = nda::basic_array_view<const value_t, Rank, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::Unified>>;

TEST(NDA, CudaConstructFromArray) {
  auto A = nda::rand<value_t>(N, N);

  // device <- host
  auto A_d = cuarray_t<2>{A};
  EXPECT_EQ(A_d.shape(), A.shape());

  // device <- device
  auto B_d = cuarray_t<2>{A_d};

  // host <- device
  auto B = array_t<2>{B_d};
  EXPECT_ARRAY_EQ(B, A);
}

TEST(NDA, CudaConstructFromView) {
  auto A = nda::rand<value_t>(N, N, N, N);

  // device <- host
  auto Av  = A(nda::range::all, 0, nda::range::all, nda::range::all);
  auto A_d = cuarray_t<3>{Av};

  // device <- device
  auto Av_d = A_d(nda::range::all, 0, nda::range::all);
  auto B_d  = cuarray_t<2>{Av_d};

  // host <- device
  auto Bv_d = B_d(nda::range::all, 0);
  auto B    = array_t<1>{Bv_d};
  EXPECT_ARRAY_EQ(B, A(nda::range::all, 0, 0, 0));
}

TEST(NDA, CudaAssignFromArray) {
  auto A = nda::rand<value_t>(N, N);

  // device <- host
  auto A_d = cuarray_t<2>(N, N);
  A_d      = A;

  // device <- device
  auto B_d = cuarray_t<2>(N, N);
  B_d      = A_d;

  // host <- device
  auto B = array_t<2>(N, N);
  B      = B_d;

  EXPECT_ARRAY_EQ(B, A);
}

TEST(NDA, CudaAssignFromView) {
  auto A = nda::rand<value_t>(N, N, N, N);

  // device <- host
  auto Av  = A(nda::range::all, 0, nda::range::all, nda::range::all);
  auto A_d = cuarray_t<3>(N, N, N);
  A_d      = Av;

  // device <- device
  auto Av_d = A_d(nda::range::all, 0, nda::range::all);
  auto B_d  = cuarray_t<2>(N, N);
  B_d       = Av_d;

  // host <- device
  auto Bv_d = B_d(nda::range::all, 0);
  auto B    = array_t<1>(N);
  B         = Bv_d;

  EXPECT_ARRAY_EQ(B, A(nda::range::all, 0, 0, 0));
}

TEST(NDA, CudaStorage) {
  auto h1      = nda::mem::handle_heap<int>{10};
  h1.data()[2] = 89;

  // device <- host
  auto h1_d = nda::heap<nda::mem::Device>::handle<int>{h1};

  // device <- host
  auto h2_d = nda::heap<nda::mem::Device>::handle<int>{h1_d};

  // host <- device
  auto h2 = nda::mem::handle_heap<int>{h2_d};

  EXPECT_EQ(h2.data()[2], 89);
}

TEST(NDA, CudaAddrSpace) {
  // compile time checks
  using namespace nda::mem;

  static_assert(on_host<array_t<1>, array_vt<2>, array_cvt<3>>, "INTERNAL");
  static_assert(not on_host<cuarray_t<1>, array_vt<2>, array_cvt<3>>, "INTERNAL");
  static_assert(not on_host<array_t<1>, unmarray_vt<2>, array_cvt<3>>, "INTERNAL");

  static_assert(on_device<cuarray_t<1>, cuarray_vt<2>, cuarray_cvt<3>>, "INTERNAL");
  static_assert(not on_device<unmarray_t<1>, cuarray_vt<2>, cuarray_cvt<3>>, "INTERNAL");
  static_assert(not on_device<cuarray_t<1>, array_vt<2>, cuarray_cvt<3>>, "INTERNAL");

  static_assert(on_unified<unmarray_t<1>, unmarray_vt<2>, unmarray_cvt<3>>, "INTERNAL");
  static_assert(not on_unified<array_t<1>, unmarray_vt<2>, unmarray_cvt<3>>, "INTERNAL");
  static_assert(not on_unified<unmarray_t<1>, cuarray_vt<2>, unmarray_cvt<3>>, "INTERNAL");

  static_assert(have_same_addr_space<array_t<1>, array_vt<2>, array_cvt<3>>, "INTERNAL");
  static_assert(have_same_addr_space<cuarray_t<1>, cuarray_vt<2>, cuarray_cvt<3>>, "INTERNAL");
  static_assert(have_same_addr_space<unmarray_t<1>, unmarray_vt<2>, unmarray_cvt<3>>, "INTERNAL");
  static_assert(not have_same_addr_space<cuarray_t<1>, array_vt<2>, array_cvt<3>>, "INTERNAL");
  static_assert(not have_same_addr_space<array_t<1>, cuarray_vt<2>, cuarray_cvt<3>>, "INTERNAL");
  static_assert(not have_same_addr_space<array_t<1>, unmarray_vt<2>, unmarray_cvt<3>>, "INTERNAL");

  static_assert(have_host_compatible_addr_space<array_t<1>, array_vt<2>, array_cvt<3>>, "INTERNAL");
  static_assert(have_host_compatible_addr_space<array_t<1>, unmarray_vt<2>, unmarray_cvt<3>>, "INTERNAL");
  static_assert(not have_host_compatible_addr_space<array_t<1>, unmarray_vt<2>, cuarray_cvt<3>>, "INTERNAL");

  static_assert(have_device_compatible_addr_space<cuarray_t<1>, cuarray_vt<2>, cuarray_cvt<3>>, "INTERNAL");
  static_assert(have_device_compatible_addr_space<cuarray_t<1>, unmarray_vt<2>, cuarray_cvt<3>>, "INTERNAL");
  static_assert(not have_device_compatible_addr_space<cuarray_t<1>, unmarray_vt<2>, array_cvt<3>>, "INTERNAL");

  static_assert(have_compatible_addr_space<cuarray_t<1>, unmarray_vt<2>, cuarray_cvt<3>>, "INTERNAL");
  static_assert(have_compatible_addr_space<array_t<1>, unmarray_vt<2>, array_cvt<3>>, "INTERNAL");
  static_assert(not have_compatible_addr_space<cuarray_t<1>, unmarray_vt<2>, array_cvt<3>>, "INTERNAL");
}
