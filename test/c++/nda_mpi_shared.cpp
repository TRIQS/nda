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
#include <atomic>

// ==============================================================

using shm_allocator = nda::mem::mpi_shm_allocator;

TEST(SHM, MoveSemantic) {
  nda::shared_array<double, 2> A;
  A.resize({4, 4});
  A(2, 2)                        = 3.1415;
  nda::shared_array<double, 2> B = std::move(A);
  EXPECT_EQ(B(2, 2), 3.1415);

  EXPECT_TRUE(A.empty());
  EXPECT_EQ(A.data(), nullptr);
  EXPECT_EQ(A.shape(), B.shape());
}

TEST(SHM, MPIFence) {
  auto shm    = shm_allocator::get_communicator();
  int my_rank = (shm.size() > 2) ? 2 : 0;
  nda::shared_array<int, 2> A({2, 2});

  if (shm.rank() == my_rank) { A(1, 1) = my_rank; }

  nda::fence(A);

  EXPECT_EQ(A(1, 1), my_rank);
}

TEST(SHM, LordOfRings) {
  auto shm = shm_allocator::get_communicator();
  int rank = shm.rank();
  int size = shm.size();

  nda::shared_array<int, 1> A(shape_t<1>{size});

  int right = (rank + 1) % size;

  A(rank) = right;

  nda::fence(A);

  for (int i = 0; i < size; i++) {
    int expected = (i + 1) % size;
    EXPECT_EQ(A(i), expected);
  }
}

TEST(SHM, Fences) {
  auto shm         = shm_allocator::get_communicator();
  int my_rank      = shm.rank();
  int size         = shm.size();
  int expected_sum = (size * (size - 1)) / 2;
  shape_t<2> shape = {3, 3};

  nda::shared_array<int, 2> A(shape);

  EXPECT_EQ(A.shape(), shape);

  for (int i = 0; i < shape[0]; ++i) {
    for (int j = 0; j < shape[1]; ++j) { A(i, j) = 0; }
  }

  nda::fence(A);

  // --- Serialized Updates ---
  // To avoid race conditions on a read-modify-write (RMW) operation (A(i,j) += my_rank),
  // we let one rank update the entire array at a time.
  for (int r = 0; r < size; r++) {
    // Only the process whose rank matches r performs the update.
    if (my_rank == r) {
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          // Each process adds its rank value to each element.
          A(i, j) += my_rank;
        }
      }
    }
    // Fence here ensures that updates from the current rank are flushed and visible
    // to all other processes before the next rank begins updating.
    nda::fence(A);
  }
  // --- End of Serialized Updates ---

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) { EXPECT_EQ(A(i, j), expected_sum); }
  }
}

TEST(SHM, FencesAtomic) {
  auto shm         = shm_allocator::get_communicator();
  int my_rank      = shm.rank();
  int size         = shm.size();
  int expected_sum = (size * (size - 1)) / 2;
  shape_t<2> shape = {3, 3};

  nda::shared_array<std::atomic<int>, 2> A(shape);

  EXPECT_EQ(A.shape(), shape);

  for (int i = 0; i < shape[0]; ++i) {
    for (int j = 0; j < shape[1]; ++j) { A(i, j) = 0; }
  }

  nda::fence(A);

  // --- Atomic Updates ---
  // Instead of serialized updates, every rank will update concurrently.
  // We assume here that A(i,j) supports an atomic fetch-add operation.
  // This pseudocode uses atomic_fetch_add which atomically adds a value
  // and returns the previous value. Replace this with your actual atomic interface.
  for (int i = 0; i < shape[0]; ++i) {
    for (int j = 0; j < shape[1]; ++j) {
      // The atomic operation ensures that concurrent updates do not conflict.
      // For example:
      std::atomic_fetch_add(&A(i, j), my_rank); // Atomically add my_rank to A(i,j)
    }
  }
  nda::fence(A);
  // --- End of Atomic Updates ---

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) { EXPECT_EQ(A(i, j), expected_sum); }
  }
}

TEST(SHM, RowSum) {
  auto shm         = shm_allocator::get_communicator();
  int rank         = shm.rank();
  int size         = shm.size();
  int expected_sum = (size * (size - 1)) / 2;
  shape_t<2> shape = {size, 10};

  nda::shared_array<int, 2> A(shape);

  for (int j = 0; j < 10; j++) { A(rank, j) = rank; }

  nda::fence(A);

  for (int j = 0; j < 10; j++) {
    int colSum = 0;
    for (int i = 0; i < size; i++) { colSum += A(i, j); }
    EXPECT_EQ(colSum, expected_sum);
  }
}

TEST(SHM, SharedArrayViewAccess) {
  auto shm = shm_allocator::get_communicator();
  nda::shared_array<int, 2> A({2, 2});
  A(1, 1)                             = 11;
  nda::shared_array_view<int, 2> view = A;
  EXPECT_EQ(view(1, 1), 11);
  nda::fence(view);
  if (shm.rank() == 0) { view(1, 1) = 99; }
  nda::fence(view);
  EXPECT_EQ(A(1, 1), 99);
}

TEST(SHM, SharedBorrowed) {
  using basic_array_borrowed_type =
     nda::basic_array<int, 2, nda::C_layout, 'A', nda::borrowed<nda::mem::MPISharedMemory, nda::mem::mpi_shm_allocator>>;
  using layout  = typename basic_array_borrowed_type::layout_t;
  using storage = typename basic_array_borrowed_type::storage_t;

  layout arr = std::array{4, 4};

  nda::mem::handle_heap<int, nda::mem::mpi_shm_allocator> h(16);
  nda::mem::handle_borrowed<int, nda::mem::MPISharedMemory, nda::mem::mpi_shm_allocator> hb(h);

  storage sto = hb;

  basic_array_borrowed_type A(arr, std::move(sto));

  A(2, 2) = 42;
  nda::fence(A);
  EXPECT_EQ(A(2, 2), 42);
}

TEST(SHM, CustomAllocator) {
  nda::mem::handle_heap<int, nda::mem::mpi_shm_allocator> h(10);
  /// TODO: check what is meant here (add concept or static assert to mallocator)
  nda::mem::handle_borrowed<int, nda::mem::AddressSpace::MPISharedMemory, nda::mem::mallocator<>> hb(h);
  EXPECT_EQ(hb.parent(), nullptr);
}

TEST(SHM, CustomAllocatorMatching) {
  nda::mem::handle_heap<int, nda::mem::mpi_shm_allocator> h(10);
  nda::mem::handle_borrowed<int, nda::mem::AddressSpace::MPISharedMemory, nda::mem::mpi_shm_allocator> hb(h);
  EXPECT_NE(hb.parent(), nullptr);
  EXPECT_EQ(h.data(), hb.data());
  EXPECT_EQ(hb.userdata(), h.userdata());
}

TEST(SHM, Concept) {
  static_assert(nda::SharedArray<nda::shared_array<int, 2>>);
  static_assert(nda::SharedArray<nda::basic_array<int, 2, nda::C_layout, 'A', nda::heap_basic<nda::mem::mpi_shm_allocator>>>);
  static_assert(!nda::SharedArray<nda::basic_array<int, 2, nda::C_layout, 'A', nda::heap<>>>);
  static_assert(!nda::SharedArray<nda::shared_array<int, 2, nda::C_layout, nda::heap<>>>);
}

TEST(SHM, Allocator) { //NOLINT
  constexpr int num_elements = 11;
  constexpr int bytes        = num_elements * sizeof(int);

  shm_allocator allocator;
  auto blk = allocator.allocate(bytes);
  int *ptr = reinterpret_cast<int *>(blk.ptr);
  EXPECT_NE(ptr, nullptr);

  for (int i = 0; i < num_elements; i++) { ptr[i] = i; }
  for (int i = 0; i < num_elements; i++) { EXPECT_EQ(ptr[i], i); }

  auto zero_blk = allocator.allocate_zero(bytes);
  int *zero_ptr = reinterpret_cast<int *>(zero_blk.ptr);

  ASSERT_NE(zero_ptr, nullptr);

  for (size_t i = 0; i < num_elements; i++) { EXPECT_EQ(zero_ptr[i], 0); }

#ifdef ADDRESS_SANITIZER
  EXPECT_DEATH(ptr[num_elements] = 42.0);
#endif

  allocator.deallocate(blk);
  allocator.deallocate(zero_blk);
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

TEST(SHM, ConstructWithShape) {
  shape_t<2> shape = {3, 3};
  nda::shared_array<int, 2> A(shape);
  EXPECT_EQ(A.shape(), shape);

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) { A(i, j) = i * 10 + j; }
  }
  fence(A);

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) { EXPECT_EQ(A(i, j), i * 10 + j); }
  }
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

TEST(SHM, ForEachChunked) {
  auto shm         = shm_allocator::get_communicator();
  int my_chunk     = shm.rank();
  int n_chunk      = shm.size();
  shape_t<2> shape = {5, 5};
  int total        = shape[0] * shape[1];

  nda::shared_array<int, 2> A(shape);

  nda::for_each_chunked([&shm](int &i) { i = shm.rank(); }, A, n_chunk, my_chunk);

  nda::fence(A);

  std::vector<int> expected(total, -1);
  for (int r = 0; r < n_chunk; r++) {
    auto chunk = itertools::chunk_range(0, total, n_chunk, r);
    for (int idx = chunk.first; idx < chunk.second; ++idx) { expected[idx] = r; }
  }

  for (int i = 0; i < total; i++) { EXPECT_EQ(A(nda::_linear_index_t{i}), expected[i]); }
}
MPI_TEST_MAIN;
