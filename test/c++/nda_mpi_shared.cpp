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
#include <version>

// ==============================================================
using mpi_shm_allocator = nda::mem::mallocator<nda::mem::MPISharedMemory>;

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
  auto shm    = nda::mem::mpi_shm::get_communicator();
  int my_rank = (shm.size() > 2) ? 2 : 0;
  nda::shared_array<int, 2> A({2, 2});

  if (shm.rank() == my_rank) { A(1, 1) = my_rank; }

  nda::fence(A);

  EXPECT_EQ(A(1, 1), my_rank);
}

TEST(SHM, LordOfRings) {
  auto shm = nda::mem::mpi_shm::get_communicator();
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
  auto shm         = nda::mem::mpi_shm::get_communicator();
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

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) { EXPECT_EQ(A(i, j), expected_sum); }
  }
}

TEST(SHM, FencesAtomic) {
#if __cpp_lib_atomic_ref
  auto shm         = nda::mem::mpi_shm::get_communicator();
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

  // Each rank atomically adds its rank value to every element in the array.
  // Although A stores plain ints, we create an atomic reference on the fly.
  for (int i = 0; i < shape[0]; ++i) {
    for (int j = 0; j < shape[1]; ++j) {
      // The atomic operation ensures that concurrent updates do not conflict.
      std::atomic_ref<int>(A(i, j)).fetch_add(my_rank); // Atomically add my_rank to A(i,j)
    }
  }
  nda::fence(A);

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) { EXPECT_EQ(A(i, j), expected_sum); }
  }
#else
  GTEST_SKIP() << "std::atomic_ref is not supported by this standard library";
#endif
}

TEST(SHM, RowSum) {
  auto shm         = nda::mem::mpi_shm::get_communicator();
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
  auto shm = nda::mem::mpi_shm::get_communicator();
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
  using basic_array_borrowed_type = nda::basic_array<int, 2, nda::C_layout, 'A', nda::borrowed<nda::mem::MPISharedMemory>>;
  using layout                    = typename basic_array_borrowed_type::layout_t;
  using storage                   = typename basic_array_borrowed_type::storage_t;

  layout arr = std::array{4, 4};

  nda::mem::handle_heap<int, mpi_shm_allocator> h(16);
  nda::mem::handle_borrowed<int, nda::mem::MPISharedMemory> hb(h);

  storage sto = hb;

  basic_array_borrowed_type A(arr, std::move(sto));

  A(2, 2) = 42;
  nda::fence(A);
  EXPECT_EQ(A(2, 2), 42);
}

TEST(SHM, MPIShmAllocator) {
  nda::mem::handle_heap<int, mpi_shm_allocator> h(10);
  nda::mem::handle_borrowed<int, nda::mem::MPISharedMemory> hb(h);
  EXPECT_NE(hb.parent(), nullptr);
}

TEST(SHM, CustomAllocatorMatching) {
  nda::mem::handle_heap<int, mpi_shm_allocator> h(10);
  nda::mem::handle_borrowed<int, nda::mem::MPISharedMemory> hb(h);
  EXPECT_NE(hb.parent(), nullptr);
  EXPECT_EQ(h.data(), hb.data());
  EXPECT_EQ(hb.userdata(), h.userdata());
}

TEST(SHM, Concept) {
  static_assert(nda::SharedArray<nda::shared_array<int, 2>>);
  static_assert(nda::SharedArray<nda::basic_array<int, 2, nda::C_layout, 'A', nda::heap_basic<mpi_shm_allocator>>>);
  static_assert(!nda::SharedArray<nda::basic_array<int, 2, nda::C_layout, 'A', nda::heap<>>>);
}

TEST(SHM, Allocator) { //NOLINT
  constexpr int num_elements = 11;
  constexpr int bytes        = num_elements * sizeof(int);

  mpi_shm_allocator allocator;
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
  nda::basic_array<long, 2, nda::C_layout, 'A', nda::heap_basic<mpi_shm_allocator>> A(3, 3);
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
  auto shm         = nda::mem::mpi_shm::get_communicator();
  int size         = shm.size();
  shape_t<2> shape = {5, 5};
  int total        = shape[0] * shape[1];

  nda::shared_array<int, 2> A(shape);

  for (auto &i : mpi::chunk(A)) {
    i = shm.rank();
  }

  nda::fence(A);

  std::vector<int> expected(total, -1);
  for (int r = 0; r < size; r++) {
    auto chunk = itertools::chunk_range(0, total, size, r);
    for (int idx = chunk.first; idx < chunk.second; ++idx) { expected[idx] = r; }
  }

  for (int i = 0; i < total; i++) { EXPECT_EQ(A(nda::_linear_index_t{i}), expected[i]); }
}

/*
TEST(MPISharedMemory, HandleHeapCopyConstructor) {
  using allocator_type = mpi_shm_allocator;
  nda::mem::handle_heap<int, allocator_type> h_original(100);

  for (int i = 0; i < h_original.size(); ++i) {
    h_original.data()[i] = i * 2; // any pattern you like
  }

  nda::mem::handle_heap<int, allocator_type> h_copy(h_original);
  EXPECT_EQ(h_original.size(), h_copy.size());
  EXPECT_NE(h_original.data(), h_copy.data());

  for (int i = 0; i < h_original.size(); ++i) {
    EXPECT_EQ(h_original.data()[i], h_copy.data()[i]);
  }

  h_original.data()[0] = -1;
  EXPECT_NE(h_original.data()[0], h_copy.data()[0]);
}
*/
MPI_TEST_MAIN;
