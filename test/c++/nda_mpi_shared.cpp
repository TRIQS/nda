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

TEST(SHM, Allocator) { //NOLINT
  nda::basic_array<long, 2, nda::C_layout, 'A', nda::mpi_shared_memory<nda::mem::mpi_shm_allocator>> A(3, 3);
  EXPECT_EQ(A.shape(), (shape_t<2>{3, 3}));

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      A(i, j) = i * 10 + j;
    }
  }

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_EQ(A(i, j), i * 10 + j);
    }
  }
}

TEST(SHM, Constructor) { //NOLINT
  mpi::communicator world;
  mpi::shared_communicator shm = world.split_shared();
  nda::shared_array<long, 2> A(shm);
}

TEST(SHM, AccessElement) {
  mpi::communicator world;
  mpi::shared_communicator shm = world.split_shared();
  nda::shared_array<int, 2> A(shm);
  A.resize({3,3});

  A(0,0) = 42;
  EXPECT_EQ(A(0,0), 42);
}

TEST(SHM, MoveSemantic) {
  mpi::communicator world;
  mpi::shared_communicator shm = world.split_shared();
  nda::shared_array<double, 2> A(shm);

  A.resize({4, 4});
  A(2, 2) = 3.1415;

  nda::shared_array<double, 2> B = std::move(A);

  EXPECT_EQ(B(2, 2), 3.1415);
  EXPECT_EQ(A.size(), 0);
}

TEST(SHM, SubArray) {
  mpi::communicator world;
  mpi::shared_communicator shm = world.split_shared();
  nda::shared_array<double, 2> A(shm);

  A.resize({4, 4});
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
      A(i, j) = i * 10 + j;

  auto sub_A = A(nda::range(1,3), nda::range(1,4));
  EXPECT_EQ(sub_A(0, 0), A(1, 1));
  EXPECT_EQ(sub_A(0, 1), A(1, 2));
  EXPECT_EQ(sub_A(0, 2), A(1, 3));
  EXPECT_EQ(sub_A(1, 0), A(2, 1));
  EXPECT_EQ(sub_A(1, 1), A(2, 2));
  EXPECT_EQ(sub_A(1, 2), A(2, 3));
}

TEST(SHM, SyncAcrossRanks) {
  mpi::communicator world;
  mpi::shared_communicator shm = world.split_shared();
  nda::shared_array<int, 2> A(shm);

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
  mpi::communicator world;
  mpi::shared_communicator shm = world.split_shared();

  std::array<long, 2> shape = {3, 3};

  nda::shared_array<int, 2> A(shm, shape);

  EXPECT_EQ(A.shape(), (shape_t<2>{3, 3}));

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      A(i, j) = i * 10 + j;
    }
  }

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_EQ(A(i, j), i * 10 + j);
    }
  }
}
/*
TEST(SHM, ParallelReadWrite) {
  mpi::communicator world;
  mpi::shared_communicator shm = world.split_shared();
  nda::shared_array<int, 2> A(shm);
  A.resize({2, 2});

  A(shm.rank(), shm.rank()) = shm.rank() + 100;

  shm.barrier();

  for (int r = 0; r < shm.size(); ++r) {
    EXPECT_EQ(A(r, r), r + 100);
  }
}
  */

MPI_TEST_MAIN;
