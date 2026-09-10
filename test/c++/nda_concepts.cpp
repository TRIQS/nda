// Copyright (c) 2020--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#include "./test_common.hpp"

#include <nda/concepts.hpp>
#include <nda/nda.hpp>

#include <cstdint>
#include <memory>
#include <vector>

// Callable with 3 longs.
struct callable {
  void operator()(long, int, char) const {}
};

// Not callable with 3 longs.
struct not_callable {};

TEST(NDA, ConceptsGeneral) {
  static_assert(nda::AnyOf<int, int, double>);
  static_assert(not nda::AnyOf<float, int, double>);
  static_assert(not nda::AnyOf<int>);

  static_assert(nda::CallableWithLongs<callable, 3>);
  static_assert(not nda::CallableWithLongs<callable, 2>);
  static_assert(not nda::CallableWithLongs<not_callable, 0>);
  static_assert(nda::CallableWithLongs<nda::array<double, 5>, 5>);

  static_assert(nda::DoubleOrComplex<double>);
  static_assert(nda::DoubleOrComplex<std::complex<double>>);
  static_assert(nda::DoubleOrComplex<std::complex<float>>);
  static_assert(not nda::DoubleOrComplex<float>);

  static_assert(nda::InstantiationOf<std::complex<double>, std::complex>);
  static_assert(nda::InstantiationOf<std::vector<int>, std::vector>);
  static_assert(not nda::InstantiationOf<int, std::vector>);

  static_assert(nda::Scalar<long>);
  static_assert(nda::Scalar<double>);
  static_assert(nda::Scalar<std::complex<float>>);
  static_assert(not nda::Scalar<std::array<int, 3>>);

  static_assert(nda::StdArrayOfLong<std::array<long, 2>>);
  static_assert(not nda::StdArrayOfLong<std::array<int, 4>>);
  static_assert(not nda::StdArrayOfLong<std::vector<long>>);

  static_assert(nda::FloatOrDouble<float>);
  static_assert(nda::FloatOrDouble<double>);
  static_assert(not nda::FloatOrDouble<int>);
  static_assert(not nda::FloatOrDouble<std::complex<double>>);
}

enum unscoped_enum { e0, e1 };

TEST(NDA, ConceptsIndexType) {
  static_assert(nda::IndexType<int>);
  static_assert(nda::IndexType<long>);
  static_assert(nda::IndexType<size_t>);
  static_assert(nda::IndexType<unsigned>);
  static_assert(nda::IndexType<int8_t>);

  static_assert(not nda::IndexType<bool>);
  static_assert(not nda::IndexType<char>);
  static_assert(not nda::IndexType<char32_t>);
  static_assert(not nda::IndexType<double>);
  static_assert(not nda::IndexType<nda::range>);
  static_assert(not nda::IndexType<nda::range::all_t>);
  static_assert(not nda::IndexType<nda::ellipsis>);
  static_assert(not nda::IndexType<unscoped_enum>);
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
