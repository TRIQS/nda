// Copyright (c) 2022--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

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

TEST(NDA, CudaFill) {
  {
    auto A   = array_t<1>(N, 2.5);
    auto A_d = cuarray_t<1>(N);
    nda::mem::fill_n<nda::mem::Device>(A_d.data(), N, A[0]);
    EXPECT_ARRAY_EQ(nda::to_host(A_d), A);

    A() = 3.5;
    nda::mem::fill<nda::mem::Device>(A_d.data(), A_d.data() + N, A[0]);
    EXPECT_ARRAY_EQ(nda::to_host(A_d), A);

    A() = 0.0;
    nda::mem::fill_n<nda::mem::Device>(A_d.data(), N, A[0]);
    EXPECT_ARRAY_EQ(nda::to_host(A_d), A);
  }

  {
    // this assumes C_stride layout
    auto A   = array_t<2>(2 * N, N);
    auto A_d = cuarray_t<2>(2 * N, N);

    A() = 5.0;
    nda::mem::fill2D_n<nda::mem::Device>(A_d.data(), A_d.strides()[0], A_d.extent(1), A_d.extent(0), A(0, 0));
    EXPECT_ARRAY_EQ(nda::to_host(A_d), A);

    auto A_v = A(nda::range(0, 2 * N, 2), nda::range::all);
    A_v()    = 1.0;
    nda::mem::fill2D_n<nda::mem::Device>(A_d.data(), A_v.strides()[0], A_v.extent(1), A_v.extent(0), A_v(0, 0));
    EXPECT_ARRAY_EQ(nda::to_host(A_d), A);
  }

  {
    // Contiguous fill
    auto A_d = cuarray_t<2>(N, N);
    A_d      = 1.0;
    EXPECT_ARRAY_EQ(to_host(A_d), nda::ones<value_t>(N, N));

    // Non-contiguous fill
    auto B_d                                   = cuarray_t<2>(N, N);
    B_d(nda::range::all, nda::range(N / 2))    = 1.0;
    B_d(nda::range::all, nda::range(N / 2, N)) = 1.0;
    EXPECT_ARRAY_EQ(to_host(B_d), nda::ones<value_t>(N, N));
  }
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
