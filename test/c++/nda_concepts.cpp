// Copyright (c) 2020--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

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

TEST(NDA, DefaultAllocator) {
  nda::mem::handle_heap<int, nda::mem::mallocator<>> h(10);

  nda::mem::handle_borrowed<int> hb(h);

  EXPECT_NE(hb.parent(), nullptr);

  EXPECT_EQ(h.data(), hb.data());
}

TEST(NDA, CustomAllocator) {
  mpi::communicator world;
  mpi::shared_communicator shm = world.split_shared();
  nda::mem::mpi_shm_allocator::init(shm);
  nda::mem::handle_heap<int, nda::mem::mpi_shm_allocator> h(10);

  nda::mem::handle_borrowed<int, nda::mem::AddressSpace::MPISharedMemory, nda::mem::mallocator<>> hb(h);

  EXPECT_EQ(hb.parent(), nullptr);
}

TEST(NDA, BorrowFromPointer) {
  int arr[5] = {1, 2, 3, 4, 5};

  nda::mem::handle_borrowed<int> hb(arr);
  EXPECT_EQ(hb.data(), arr);

  EXPECT_EQ(hb.parent(), nullptr);
}

TEST(NDA, BorrowWithOffset) {
  nda::mem::handle_heap<int, nda::mem::mallocator<>> h(10);

  nda::mem::handle_borrowed<int> hb(h, 2);
  EXPECT_EQ(hb.data(), h.data() + 2);

  EXPECT_NE(hb.parent(), nullptr);
}

TEST(NDA, CustomAllocatorMatching) {
  mpi::communicator world;
  mpi::shared_communicator shm = world.split_shared();
  nda::mem::mpi_shm_allocator::init(shm);
  nda::mem::handle_heap<int, nda::mem::mpi_shm_allocator> h(10);

  nda::mem::handle_borrowed<int, nda::mem::AddressSpace::MPISharedMemory, nda::mem::mpi_shm_allocator> hb(h);

  EXPECT_NE(hb.parent(), nullptr);
  EXPECT_EQ(h.data(), hb.data());
}
