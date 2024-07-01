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
//
// Authors: Olivier Parcollet, Nils Wentzell

#include "./test_common.hpp"

#include <nda/concepts.hpp>
#include <nda/nda.hpp>

#include <memory>
#include <vector>

// Callable with 3 longs.
struct callable {
  void operator()(long, int, char) const {}
};

// Not callable with 3 longs.
struct not_callable {};

TEST(NDA, ConceptsGeneral) {
  static_assert(nda::CallableWithLongs<callable, 3>);
  static_assert(not nda::CallableWithLongs<callable, 2>);
  static_assert(not nda::CallableWithLongs<not_callable, 0>);
  static_assert(nda::CallableWithLongs<nda::array<double, 5>, 5>);

  static_assert(nda::StdArrayOfLong<std::array<long, 2>>);
  static_assert(not nda::StdArrayOfLong<std::array<int, 4>>);
  static_assert(not nda::StdArrayOfLong<std::vector<long>>);
}

TEST(NDA, ConceptsNDASpecific) {
  static_assert(nda::mem::Allocator<nda::mem::mallocator<>>);
  static_assert(nda::mem::Allocator<nda::mem::bucket<10>>);
  static_assert(nda::mem::Allocator<nda::mem::multi_bucket<2>>);
  static_assert(nda::mem::Allocator<nda::mem::segregator<1000, nda::mem::mallocator<>, nda::mem::bucket<10>>>);
  static_assert(not nda::mem::Allocator<std::allocator<int>>);

  static_assert(nda::mem::Handle<nda::mem::handle_heap<double>>);
  static_assert(nda::mem::Handle<nda::mem::handle_stack<double, 1024>>);
  static_assert(nda::mem::Handle<nda::mem::handle_sso<double, 1024>>);
  static_assert(nda::mem::Handle<nda::mem::handle_shared<double, nda::mem::AddressSpace::Device>>);
  static_assert(nda::mem::Handle<nda::mem::handle_borrowed<double>>);

  static_assert(nda::mem::OwningHandle<nda::mem::handle_heap<double>>);
  static_assert(not nda::mem::OwningHandle<nda::mem::handle_heap<const double>>);
  static_assert(not nda::mem::OwningHandle<nda::mem::handle_borrowed<double>>);

  static_assert(nda::Array<nda::array<int, 2>>);
  static_assert(nda::Array<nda::array_view<int, 2>>);
  static_assert(not nda::Array<callable>);

  static_assert(nda::MemoryArray<nda::array<int, 2>>);
  static_assert(nda::MemoryArray<nda::array_view<int, 2>>);

  static_assert(nda::ArrayOfRank<nda::array<int, 2>, 2>);
  static_assert(not nda::ArrayOfRank<nda::array<int, 2>, 1>);
  static_assert(not nda::ArrayOfRank<nda::array<int, 2>, 3>);
  static_assert(nda::ArrayOfRank<nda::array_view<int, 2>, 2>);
  static_assert(not nda::ArrayOfRank<nda::array_view<int, 2>, 1>);
  static_assert(not nda::ArrayOfRank<nda::array_view<int, 2>, 3>);

  static_assert(nda::Matrix<nda::array<int, 2>>);
  static_assert(nda::Matrix<nda::array_view<int, 2>>);
  static_assert(not nda::Matrix<nda::array<int, 1>>);
  static_assert(not nda::Matrix<nda::array_view<int, 3>>);

  static_assert(nda::Vector<nda::array<int, 1>>);
  static_assert(nda::Vector<nda::array_view<int, 1>>);
  static_assert(not nda::Vector<nda::array<int, 2>>);
  static_assert(not nda::Vector<nda::array_view<int, 3>>);

  static_assert(nda::HasValueTypeConstructibleFrom<nda::array<int, 2>, std::complex<double>>);
  static_assert(not nda::HasValueTypeConstructibleFrom<nda::array<std::complex<double>, 2>, int>);
}
