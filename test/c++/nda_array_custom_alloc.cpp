// Copyright (c) 2020-2022 Simons Foundation
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

#include <nda/nda.hpp>

TEST(NDA, ArrayWithCustomAllocator) {
  using alloc_t = nda::mem::segregator<8ul * 100, nda::mem::multi_bucket<8 * 100>, nda::mem::mallocator<>>;
  nda::basic_array<long, 2, nda::C_layout, 'A', nda::heap_basic<alloc_t>> A(3, 3);
  EXPECT_EQ(A.shape(), (shape_t<2>{3, 3}));
}

TEST(NDA, SSOArray) {
  nda::basic_array<long, 2, nda::C_layout, 'A', nda::sso<10>> A(3, 3);
  EXPECT_EQ(A.shape(), (shape_t<2>{3, 3}));
  EXPECT_FALSE(A.storage().on_heap());

  nda::basic_array<long, 2, nda::C_layout, 'A', nda::sso<10>> B(3, 4);
  EXPECT_EQ(B.shape(), (shape_t<2>{3, 4}));
  EXPECT_TRUE(B.storage().on_heap());
}
