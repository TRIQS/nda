// Copyright (c) 2019-2023 Simons Foundation
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

// alone because it is quite long to run ... (exception ....)

TEST(Array, BoundCheck) { //NOLINT

  nda::array<long, 2> A(2, 3);

  EXPECT_THROW(A(0, 3), std::runtime_error); //NOLINT

  EXPECT_THROW(A(nda::range(0, 4), 2), std::runtime_error);   //NOLINT
  EXPECT_THROW(A(nda::range(10, 14), 2), std::runtime_error); //NOLINT

  EXPECT_THROW(A(nda::range::all, 5), std::runtime_error); //NOLINT
  EXPECT_THROW(A(0, 3), std::runtime_error);               //NOLINT
}
