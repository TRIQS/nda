// Copyright (c) 2020 Simons Foundation
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

#include "./test_common.hpp"

// ==============================================================

using namespace nda;

using value_t   = double;
constexpr int N = 4;

template <size_t Rank>
using cuarray_t = nda::cuarray<value_t, Rank>;

template <size_t Rank>
using array_t = nda::array<value_t, Rank>;

template <size_t Rank>
using array_cvt = nda::array_view<const value_t, Rank>;

TEST(Cuda, ConstructFromArray) { //NOLINT
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

TEST(Cuda, ConstructFromView) { //NOLINT
  auto A = nda::rand<value_t>(N, N, N, N);

  // device <- host
  auto Av  = A(_, 0, _, _);
  auto A_d = cuarray_t<3>{Av};

  // device <- device
  auto Av_d = A_d(_, 0, _);
  auto B_d  = cuarray_t<2>{Av_d};

  // host <- device
  auto Bv_d = B_d(_, 0);
  auto B    = array_t<1>{Bv_d};
  EXPECT_ARRAY_EQ(B, A(_, 0, 0, 0));
}

TEST(Cuda, AssignFromArray) { //NOLINT
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

TEST(Cuda, AssignFromView) { //NOLINT
  auto A = nda::rand<value_t>(N, N, N, N);

  // device <- host
  auto Av  = A(_, 0, _, _);
  auto A_d = cuarray_t<3>(N, N, N);
  A_d      = Av;

  // device <- device
  auto Av_d = A_d(_, 0, _);
  auto B_d  = cuarray_t<2>(N, N);
  B_d       = Av_d;

  // host <- device
  auto Bv_d = B_d(_, 0);
  auto B    = array_t<1>(N);
  B         = Bv_d;

  EXPECT_ARRAY_EQ(B, A(_, 0, 0, 0));
}

#include <nda/mem/handle.hpp>
TEST(Cuda, Storage) { // NOLINT
  using namespace nda::mem;

  auto h1      = handle_heap<int>{10};
  h1.data()[2] = 89;

  // device <- host
  auto h1_d = heap<mem::Device>::handle<int>{h1};

  // device <- host
  auto h2_d = heap<mem::Device>::handle<int>{h1_d};

  // host <- device
  auto h2 = handle_heap<int>{h2_d};

  EXPECT_EQ(h2.data()[2], 89); //NOLINT
}
