// Copyright (c) 2020-2023 Simons Foundation
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

#define NDA_DEBUG_LEAK_CHECK

#include "./test_common.hpp"

#include <nda/basic_array.hpp>
#include <nda/mem.hpp>

// ==============================================================

TEST(SHM, Allocator) { //NOLINT
  nda::basic_array<long, 2, nda::C_layout, 'A', nda::shared_memory<nda::mem::shared_allocator>> A(3, 3);
  EXPECT_EQ(A.shape(), (shape_t<2>{3, 3}));
}

MPI_TEST_MAIN;
