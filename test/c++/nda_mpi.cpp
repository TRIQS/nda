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

#include <nda/gtest_tools.hpp>
#include <nda/nda.hpp>
#include <nda/mpi.hpp>

#include <itertools/itertools.hpp>
#include <mpi/mpi.hpp>

#include <numeric>

// Test fixture for testing various algorithms.
struct NDAMpi : public ::testing::Test {
  protected:
  NDAMpi() {
    A.resize(shape_3d);
    std::iota(A.begin(), A.end(), 0);
    M = nda::matrix<std::complex<double>>::zeros(shape_2d);
    for (long i = 0; i < shape_2d[0]; ++i) {
      for (long j = 0; j < shape_2d[1]; ++j) {
        auto x  = static_cast<double>(i * shape_2d[1] + j);
        M(i, j) = std::complex<double>(x, x + 1.0);
      }
    }
  }

  std::array<long, 3> shape_3d{6, 4, 2};
  nda::array<long, 3> A;
  std::array<long, 2> shape_2d{4, 4};
  nda::matrix<std::complex<double>> M;
  const int root = 0;
  mpi::communicator comm;
};

TEST_F(NDAMpi, Broadcast) {
  // broadcast to arrays with same dimensions
  auto A_bcast = A;
  if (comm.rank() != root) {
    A_bcast = 0;
    EXPECT_ARRAY_ZERO(A_bcast);
  } else {
    EXPECT_ARRAY_EQ(A, A_bcast);
  }
  mpi::broadcast(A_bcast, comm, root);
  EXPECT_ARRAY_EQ(A, A_bcast);

  // broadcast to arrays with different dimensions
  decltype(A) B_bcast;
  if (comm.rank() != root) {
    EXPECT_NE(A.shape(), B_bcast.shape());
  } else {
    B_bcast = A;
    EXPECT_ARRAY_EQ(A, B_bcast);
  }
  mpi::broadcast(B_bcast, comm, root);
  EXPECT_ARRAY_EQ(A, B_bcast);

  // broadcast a matrix into an array view
  if (comm.rank() != root) {
    auto C_bcast = nda::array<std::complex<double>, 3>::zeros(2, shape_2d[0], shape_2d[1]);
    EXPECT_ARRAY_ZERO(C_bcast);
    auto C_view = C_bcast(1, nda::ellipsis{});
    mpi::broadcast(C_view, comm, root);
    EXPECT_ARRAY_EQ(M, C_view);
    EXPECT_ARRAY_ZERO(C_bcast(0, nda::ellipsis{}));
  } else {
    mpi::broadcast(M);
  }
}

TEST_F(NDAMpi, Gather) {
  // all gather an array
  auto B               = nda::make_regular(A * (comm.rank() + 1));
  decltype(B) B_gather = mpi::all_gather(B, comm);
  EXPECT_EQ(B_gather.shape()[0], comm.size() * B.shape()[0]);
  for (int i = 0; i < comm.size(); ++i) {
    auto view = B_gather(nda::range(i * B.shape()[0], (i + 1) * B.shape()[0]), nda::range::all, nda::range::all);
    EXPECT_ARRAY_EQ(nda::make_regular(A * (i + 1)), view);
  }
}

TEST_F(NDAMpi, Reduce) {
  // reduce an array
  decltype(A) A_sum = mpi::reduce(A, comm);
  if (comm.rank() == 0) { EXPECT_ARRAY_EQ(nda::make_regular(A * comm.size()), A_sum); }

  // all reduce an array view
  auto B                    = nda::make_regular(A * (comm.rank() + 1));
  nda::array<long, 1> B_max = mpi::all_reduce(B(0, 0, nda::range::all), comm, MPI_MAX);
  nda::array<long, 1> B_min = mpi::all_reduce(B(0, 0, nda::range::all), comm, MPI_MIN);
  EXPECT_ARRAY_EQ(B_max, A(0, 0, nda::range::all) * comm.size());
  EXPECT_ARRAY_EQ(B_min, A(0, 0, nda::range::all));
}

TEST_F(NDAMpi, ReduceCustomType) {
  using namespace nda::clef::literals;

  // reduce an array of matrices
  using matrix_t = nda::matrix<double>;
  nda::array<matrix_t, 1> B(7);
  nda::array<matrix_t, 1> exp_sum(7);

  for (int i = 0; i < B.extent(0); ++i) {
    B(i) = matrix_t{4, 4};
    B(i)(k_, l_) << i * (comm.rank() + 1) * (k_ + l_);

    exp_sum(i) = matrix_t{4, 4};
    exp_sum(i)(k_, l_) << i * (comm.size() + 1) * comm.size() / 2 * (k_ + l_);
  }

  nda::array<matrix_t, 1> B_sum = mpi::all_reduce(B, comm);

  EXPECT_ARRAY_EQ(B_sum, exp_sum);
}

TEST_F(NDAMpi, Scatter) {
  // scatter an array
  decltype(A) A_scatter = mpi::scatter(A, comm);
  auto chunked_rg       = itertools::chunk_range(0, A.shape()[0], comm.size(), comm.rank());
  auto exp_shape        = A.shape();
  exp_shape[0]          = chunked_rg.second - chunked_rg.first;
  EXPECT_EQ(exp_shape, A_scatter.shape());
  EXPECT_ARRAY_EQ(A(nda::range(chunked_rg.first, chunked_rg.second), nda::ellipsis{}), A_scatter);
}

TEST_F(NDAMpi, BroadcastTransposedMatrix) {
  nda::matrix<std::complex<double>> M_t = transpose(M);
  nda::matrix<std::complex<double>> N;
  if (comm.rank() == 0) N = M_t;
  mpi::broadcast(N, comm, 0);
  EXPECT_ARRAY_EQ(M_t, N);
}

TEST_F(NDAMpi, BroadcastTransposedArray) {
  nda::array<long, 3> A_t = transpose(A);
  nda::array<long, 3> B(2, 4, 6);
  if (comm.rank() == 0) B = A_t;
  mpi::broadcast(B, comm, 0);
  EXPECT_ARRAY_EQ(A_t, B);
}

TEST_F(NDAMpi, VariousCollectiveCommunications) {
  using arr_t = nda::array<std::complex<double>, 2>;

  arr_t A(7, 3);
  for (int i = 0; i < A.extent(0); ++i)
    for (int j = 0; j < A.extent(1); ++j) A(i, j) = i + 10 * j;

  // scatter an array
  arr_t B;
  B       = mpi::scatter(A, comm);
  arr_t C = mpi::scatter(A, comm);
  auto rg = itertools::chunk_range(0, 7, comm.size(), comm.rank());
  EXPECT_ARRAY_EQ(B, A(nda::range(rg.first, rg.second), nda::range::all));
  EXPECT_ARRAY_NEAR(C, B);

  // gather an array
  B *= -1;
  arr_t D = mpi::gather(B, comm);
  if (comm.rank() == 0) { EXPECT_ARRAY_NEAR(D, -A); }

  // broadcast an array
  mpi::broadcast(D, comm);
  EXPECT_ARRAY_NEAR(D, -A);

  // all gather an array
  D() = 0;
  D   = mpi::all_gather(B, comm);
  EXPECT_ARRAY_NEAR(D, -A);

  // reduce an array
  arr_t R1 = mpi::reduce(A, comm);
  if (comm.rank() == 0) { EXPECT_ARRAY_NEAR(R1, comm.size() * A); }

  // all reduce an array
  arr_t R2 = mpi::all_reduce(A, comm);
  EXPECT_ARRAY_NEAR(R2, comm.size() * A);
}

MPI_TEST_MAIN
