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
#include <nda/shared_array.hpp>
#include <nda/mem.hpp>

// ==============================================================

TEST(SHM, SharedArrayAllocation) {
  nda::shared_array<int, 2> A({2, 2});

  EXPECT_EQ(A.shape(), (shape_t<2>{2, 2}));
  EXPECT_NO_THROW(A(0, 0) = 5);
  EXPECT_EQ(A(0, 0), 5);
}

TEST(SHM, MPIFence) {
  nda::shared_array<int, 2> A({2, 2});

  A(0, 0) = 42;

  nda::fence(A);
  EXPECT_EQ(A(0, 0), 42);
}

TEST(SHM, SharedArrayViewAccess) {
  nda::shared_array<int, 2> A({2, 2});
  A(1, 1) = 11;

  nda::fence(A);
  EXPECT_EQ(A(1, 1), 11);

  nda::shared_array_view<int, 2> view = A;

  EXPECT_EQ(view(1, 1), 11);
  nda::fence(A);
  view(1, 1) = 99;

  EXPECT_EQ(A(1, 1), 99);
}

TEST(SHM, ViewSync) {
  nda::shared_array<int, 2> A({2, 2});
  nda::shared_array_view<int, 2> view = A;

  view(1, 1) = 5;

  nda::fence(A);

  EXPECT_EQ(view(1, 1), 5);
}

// Test with borrowed handle?

// -------------------------

TEST(SHM, Concept) {
  static_assert(nda::SharedArray<nda::shared_array<int, 2>>);
  static_assert(nda::SharedArray<nda::basic_array<int, 2, nda::C_layout, 'A', nda::heap_basic<nda::mem::mpi_shm_allocator>>>);
  static_assert(!nda::SharedArray<nda::basic_array<int, 2, nda::C_layout, 'A', nda::heap<>>>);
  static_assert(!nda::SharedArray<nda::shared_array<int, 2, nda::C_layout, nda::heap<>>>);
}

TEST(SHM, Allocator) { //NOLINT
  nda::mem::mpi_shm_allocator allo;
  auto blk = allo.allocate(10 * sizeof(double));
  allo.deallocate(blk);
}

TEST(SHM, SimpleArray) { //NOLINT
  nda::basic_array<long, 2, nda::C_layout, 'A', nda::heap_basic<nda::mem::mpi_shm_allocator>> A(3, 3);
  EXPECT_EQ(A.shape(), (shape_t<2>{3, 3}));

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) { A(i, j) = i * 10 + j; }
  }

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) { EXPECT_EQ(A(i, j), i * 10 + j); }
  }
}

TEST(SHM, AccessElement) {
  nda::shared_array<int, 2> A;
  A.resize({3, 3});

  A(0, 0) = 42;
  EXPECT_EQ(A(0, 0), 42);
}

TEST(SHM, MoveSemantic) {
  nda::shared_array<double, 2> A;

  A.resize({4, 4});
  A(2, 2) = 3.1415;

  nda::shared_array<double, 2> B = std::move(A);

  EXPECT_EQ(B(2, 2), 3.1415);
}

TEST(SHM, SubArray) {
  nda::shared_array<double, 2> A;

  A.resize({4, 4});
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++) A(i, j) = i * 10 + j;

  auto sub_A = A(nda::range(1, 3), nda::range(1, 4));
  EXPECT_EQ(sub_A(0, 0), A(1, 1));
  EXPECT_EQ(sub_A(0, 1), A(1, 2));
  EXPECT_EQ(sub_A(0, 2), A(1, 3));
  EXPECT_EQ(sub_A(1, 0), A(2, 1));
  EXPECT_EQ(sub_A(1, 1), A(2, 2));
  EXPECT_EQ(sub_A(1, 2), A(2, 3));
}

TEST(SHM, SyncAcrossRanks) {
  auto shm = nda::mem::mpi_shm_allocator::get_communicator();
  nda::shared_array<int, 2> A;

  A.resize({2, 2});

  if (shm.rank() == 0) {
    A(0, 0) = 42;
    A(1, 1) = 99;
  }

  shm.barrier();

  EXPECT_EQ(A(0, 0), 42);
  EXPECT_EQ(A(1, 1), 99);
}

TEST(SHM, ConstructWithShape) {
  auto shm         = nda::mem::mpi_shm_allocator::get_communicator();
  shape_t<2> shape = {3, 3};
  nda::shared_array<int, 2> A(shape);

  EXPECT_EQ(A.shape(), shape);

  if (shm.rank() == 0) {
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) { A(i, j) = i * 10 + j; }
    }
  }
  fence(A);

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) { EXPECT_EQ(A(i, j), i * 10 + j); }
  }
}

/*
TEST(SHM, Fences) {
  mpi::communicator world;
  mpi::shared_communicator shm = world.split_shared();

  shape_t<2> shape = {3, 3};

  nda::shared_array<int, 2> A(shape);

  EXPECT_EQ(A.shape(), shape);

  fence(A);

  if (shm.rank() == 0) {
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) { A(i, j) = 0; }
    }
  }

  fence(A);

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) { A(i, j) += shm.rank(); }
  }

  fence(A);

  int sum = 0;
  for (int r = 0; r < shm.size(); ++r) { sum += r; }

  shm.barrier();

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) { EXPECT_EQ(A(i, j), sum); }
  }
}
*/

/*
TEST(SHM, ForEachChunked) {
  mpi::communicator world;
  mpi::shared_communicator shm = world.split_shared();

  shape_t<2> shape = {3, 3};

  nda::shared_array<int, 2> A(shape);

  int n_chunks = shm.size();
  int my_chunk = shm.rank();

  nda::for_each_chunked([&shm](int &i) { i = shm.rank(); }, A, n_chunks, my_chunk);

  int total_elements = shape[0] * shape[1];

  int base_count = total_elements / n_chunks;
  int remainder = total_elements % n_chunks;

  auto expected_for_index = [=](int k) -> int {
    int start = 0;
    for (int i = 0; i < n_chunks; i++) {
      int count = base_count + (i < remainder ? 1 : 0);
      if (k < start + count) {
        return i;
      }
      start += count;
    }
    return -1;
  };

  for (int i = 0; i < 3; ++i) {
    std::cout << "[rank " << world.rank() << "] ";
    for (int j = 0; j < 3; ++j) {
      std::cout << A(i,j) << " ";
    }
    std::cout << "\n";
  }
}
*/
MPI_TEST_MAIN;
