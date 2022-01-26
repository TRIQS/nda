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

using value_t      = double;
constexpr int N    = 4;
constexpr int Rank = 2;

using namespace nda;

using device_array_t = nda::basic_array<value_t, Rank, C_layout, 'A', nda::heap<mem::Device>>;
using array_t        = nda::array<value_t, Rank>;
using array_cvt      = nda::array_view<const value_t, Rank>;

TEST(Cuda, Construct) { //NOLINT
  auto A = nda::rand<value_t>(N, N);

  // device <- host
  auto A_d = device_array_t{A};
  EXPECT_EQ(A_d.shape(), A.shape());

  // host <- device
  auto B = array_t{A_d};
  EXPECT_ARRAY_EQ(B, A);
}

TEST(Cuda, Assign) { //NOLINT
  auto A = nda::rand<value_t>(N, N);

  // device <- host
  auto A_d = device_array_t(N, N);
  A_d      = A;

  // host <- device
  auto B = array_t(N, N);
  B      = A_d;

  EXPECT_ARRAY_EQ(B, A);
}

#include <nda/mem/handle.hpp>
TEST(Cuda, Storage) { // NOLINT
  using namespace nda::mem;

  auto h1      = handle_heap<int>{10};
  h1.data()[2] = 89;

  // device <- host
  auto h_d = heap<mem::Device>::handle<int>{h1};

  // host <- device
  auto h2 = handle_heap<int>{h_d};

  EXPECT_EQ(h2.data()[2], 89); //NOLINT
}
