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

// just compile time

TEST(NDA, Concept) { // NOLINT

  static_assert(!std::is_pod<nda::array<long, 2>>::value, "POD pb");
  static_assert(nda::is_scalar_for_v<int, matrix<std::complex<double>>> == 1, "oops");

#if __cplusplus > 201703L

  using nda::Array;
  using nda::ArrayOfRank;

  static_assert(Array<nda::array<int, 2>>, "INTERNAL");
  static_assert(ArrayOfRank<nda::array<int, 2>, 2>, "INTERNAL");
  static_assert(not ArrayOfRank<nda::array<int, 2>, 1>, "INTERNAL");
  static_assert(not ArrayOfRank<nda::array<int, 2>, 3>, "INTERNAL");

  static_assert(Array<nda::array_view<int, 2>>, "INTERNAL");
  static_assert(ArrayOfRank<nda::array_view<int, 2>, 2>, "INTERNAL");
  static_assert(not ArrayOfRank<nda::array_view<int, 2>, 1>, "INTERNAL");
  static_assert(not ArrayOfRank<nda::array_view<int, 2>, 3>, "INTERNAL");

#endif
}
