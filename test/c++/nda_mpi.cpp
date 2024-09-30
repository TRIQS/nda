// Copyright (c) 2019-2024 Simons Foundation
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
// Authors: Thomas Hahn, Olivier Parcollet, Nils Wentzell

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
    A2 = A;
    M2 = M;
  }

  std::array<long, 3> shape_3d{6, 4, 2};
  nda::array<long, 3> A;
  nda::array<long, 3, nda::basic_layout<0, nda::encode(std::array{1, 2, 0}), nda::layout_prop_e::contiguous>> A2;
  std::array<long, 2> shape_2d{4, 4};
  nda::matrix<std::complex<double>> M;
  nda::matrix<std::complex<double>, nda::F_layout> M2;
  const int root         = 0;
  mpi::communicator comm = {};
  long mpi_size          = comm.size();
  long mpi_rank          = comm.rank();
  nda::range::all_t _    = {};
};

TEST_F(NDAMpi, ExpectEqualRanks) {
  // test the expect_equal_ranks function
  if (mpi_size > 1) {
    EXPECT_TRUE(nda::detail::have_mpi_equal_ranks(A, comm));
    if (mpi_rank == 1) {
      auto B = nda::array<long, 2>::zeros(6, 4);
      EXPECT_FALSE(nda::detail::have_mpi_equal_ranks(B, comm));
    } else {
      EXPECT_FALSE(nda::detail::have_mpi_equal_ranks(A, comm));
    }
  }
}

TEST_F(NDAMpi, ExpectEqualShapes) {
  // test the expect_equal_shapes function
  if (mpi_size > 1) {
    EXPECT_TRUE(nda::detail::have_mpi_equal_shapes(A, comm));
    if (mpi_rank == 1) A = nda::array<long, 3>::zeros(6, 5, 2);
    EXPECT_FALSE(nda::detail::have_mpi_equal_shapes(A, comm));
  }
}

TEST_F(NDAMpi, ExpectEqualShapeSaveFirst) {
  // test the expect_equal_shape_save_first function
  if (mpi_size > 1) {
    if (mpi_rank == 1) A = nda::array<long, 3>::zeros(5, 4, 2);
    EXPECT_TRUE(nda::detail::have_mpi_equal_shapes(A(nda::range{1}, nda::ellipsis{}), comm));
    if (mpi_rank == 1) A = nda::array<long, 3>::zeros(5, 5, 2);
    EXPECT_FALSE(nda::detail::have_mpi_equal_shapes(A(nda::range{1}, nda::ellipsis{}), comm));
  }
}

TEST_F(NDAMpi, BroadcastCLayout) {
  // broadcast to arrays with same dimensions
  auto A_bcast = A;
  if (mpi_rank == root) {
    EXPECT_ARRAY_EQ(A, A_bcast);
  } else {
    A_bcast = 0;
    EXPECT_ARRAY_ZERO(A_bcast);
  }
  mpi::broadcast(A_bcast, comm, root);
  EXPECT_ARRAY_EQ(A, A_bcast);

  // broadcast to arrays with different dimensions
  decltype(A) B_bcast;
  if (mpi_rank == root) {
    B_bcast = A;
    EXPECT_ARRAY_EQ(A, B_bcast);
  } else {
    EXPECT_NE(A.shape(), B_bcast.shape());
  }
  mpi::broadcast(B_bcast, comm, root);
  EXPECT_ARRAY_EQ(A, B_bcast);

  // broadcast a matrix into an array view
  if (mpi_rank == root) {
    mpi::broadcast(M, comm, root);
  } else {
    auto C_bcast = nda::array<std::complex<double>, 3>::zeros(2, shape_2d[0], shape_2d[1]);
    EXPECT_ARRAY_ZERO(C_bcast);
    auto C_view = C_bcast(1, nda::ellipsis{});
    mpi::broadcast(C_view, comm, root);
    EXPECT_ARRAY_EQ(M, C_view);
    EXPECT_ARRAY_ZERO(C_bcast(0, nda::ellipsis{}));
  }

  // broadcast a view to views
  if (mpi_rank == root) {
    mpi::broadcast(M(0, _), comm, root);
  } else {
    auto M_bcast = M;
    M_bcast      = 0;
    mpi::broadcast(M_bcast(0, _), comm, root);
    EXPECT_ARRAY_ZERO(M_bcast(nda::range(1, shape_2d[0]), _));
    EXPECT_ARRAY_EQ(M_bcast(0, _), M(0, _));
  }
}

TEST_F(NDAMpi, BroadcastOtherLayouts) {
  // broadcast to arrays with same dimensions
  auto A2_bcast = A2;
  if (mpi_rank == root) {
    EXPECT_ARRAY_EQ(A2, A2_bcast);
  } else {
    A2_bcast = 0;
    EXPECT_ARRAY_ZERO(A2_bcast);
  }
  mpi::broadcast(A2_bcast, comm, root);
  EXPECT_ARRAY_EQ(A2, A2_bcast);

  // broadcast to arrays with different dimensions
  decltype(A2) B2_bcast;
  if (mpi_rank == root) {
    B2_bcast = A2;
    EXPECT_ARRAY_EQ(A2, B2_bcast);
  } else {
    EXPECT_NE(A2.shape(), B2_bcast.shape());
  }
  mpi::broadcast(B2_bcast, comm, root);
  EXPECT_ARRAY_EQ(A2, B2_bcast);

  // broadcast a matrix into an array view
  if (mpi_rank == root) {
    mpi::broadcast(M2, comm, root);
  } else {
    auto C2_bcast = nda::array<std::complex<double>, 3, nda::F_layout>::zeros(shape_2d[0], shape_2d[1], 2);
    EXPECT_ARRAY_ZERO(C2_bcast);
    auto C2_view = C2_bcast(nda::ellipsis{}, 1);
    mpi::broadcast(C2_view, comm, root);
    EXPECT_ARRAY_EQ(M2, C2_view);
    EXPECT_ARRAY_ZERO(C2_bcast(nda::ellipsis{}, 0));
  }

  // broadcast a view to views
  if (mpi_rank == root) {
    mpi::broadcast(M2(_, 0), comm, root);
  } else {
    auto M2_bcast = M2;
    M2_bcast      = 0;
    mpi::broadcast(M2_bcast(_, 0), comm, root);
    EXPECT_ARRAY_ZERO(M2_bcast(_, nda::range(1, shape_2d[1])));
    EXPECT_ARRAY_EQ(M2_bcast(_, 0), M2(_, 0));
  }

  // broadcast a C layout matrix into an F layout matrix
  if (mpi_rank == root) {
    mpi::broadcast(M, comm, root);
  } else {
    mpi::broadcast(M2, comm, root);
    EXPECT_ARRAY_EQ(nda::transpose(M), M2);
  }
}

TEST_F(NDAMpi, Gather1DArray) {
  // allgather 1-dimensional arrays of different sizes
  auto C        = nda::vector<int>(mpi_rank + 1, mpi_rank);
  auto C_gather = mpi::all_gather(C, comm);
  EXPECT_EQ(C_gather.size(), mpi_size * (mpi_size + 1) / 2);
  for (int ofs = 0, r = 0; r < mpi_size; ofs += ++r) {
    auto view = C_gather(nda::range(ofs, ofs + r + 1));
    auto exp  = nda::vector<int>(r + 1, r);
    EXPECT_ARRAY_EQ(exp, view);
  }
}

TEST_F(NDAMpi, GatherCLayout) {
  // allgather C-layout arrays
  auto B            = nda::make_regular(A * (mpi_rank + 1));
  auto B_gather     = mpi::all_gather(B, comm);
  auto [d0, d1, d2] = shape_3d;
  EXPECT_EQ(B_gather.shape(), (std::array{mpi_size * d0, d1, d2}));
  for (int r = 0; r < mpi_size; ++r) {
    auto view = B_gather(nda::range(r * d0, (r + 1) * d0), _, _);
    EXPECT_ARRAY_EQ(nda::make_regular(A * (r + 1)), view);
  }

  // gather C-layout matrix views
  auto rg       = nda::range(2);
  auto N        = nda::make_regular(M * (mpi_rank + 1));
  auto N_gather = mpi::gather(N(rg, _), comm);
  if (mpi_rank == root) {
    EXPECT_EQ(N_gather.shape(), (std::array<long, 2>{mpi_size * 2l, shape_2d[1]}));
    for (long r = 0; r < mpi_size; ++r) {
      auto view = N_gather(nda::range(r * 2, (r + 1) * 2), _);
      auto exp  = nda::make_regular(M * (r + 1));
      EXPECT_ARRAY_EQ(exp(rg, _), view);
    }
  } else {
    EXPECT_TRUE(N_gather.empty());
    EXPECT_TRUE(N_gather.size() == 0);
  }
}

TEST_F(NDAMpi, GatherOtherLayouts) {
  // allgather non C-layout arrays by first reshaping it
  constexpr auto perm     = decltype(A2)::layout_t::stride_order;
  constexpr auto inv_perm = nda::permutations::inverse(perm);

  decltype(A2) B2  = nda::make_regular(A2 * (mpi_rank + 1));
  auto B2_gather   = mpi::all_gather(nda::permuted_indices_view<nda::encode(inv_perm)>(B2), comm);
  auto B2_gather_v = nda::permuted_indices_view<nda::encode(perm)>(B2_gather);
  for (int r = 0; r < mpi_size; ++r) {
    auto view = B2_gather_v(_, nda::range(r * shape_3d[1], (r + 1) * shape_3d[1]), _);
    EXPECT_ARRAY_EQ(nda::make_regular(A2 * (r + 1)), view);
  }
}

TEST_F(NDAMpi, GatherCustomType) {
  // allgather an array of matrices
  using matrix_t = nda::matrix<int>;
  nda::vector<matrix_t> B(2);
  nda::vector<matrix_t> exp(2 * mpi_size);
  for (int r = 0; r < mpi_size; ++r) {
    exp(r * 2)     = matrix_t::ones(shape_2d) * r;
    exp(r * 2 + 1) = matrix_t::ones(shape_2d) * (r + 1);
    if (r == mpi_rank) {
      B(0) = exp(r * 2);
      B(1) = exp(r * 2 + 1);
    }
  }

  auto B_gathered = mpi::all_gather(B, comm);
  EXPECT_EQ(B_gathered.shape(), std::array{2l * mpi_size});
  for (int i = 0; i < B_gathered.size(); ++i) { EXPECT_ARRAY_EQ(exp(i), B_gathered(i)); }
}

TEST_F(NDAMpi, LazyGather) {
  // lazy-allgather 1-dimensional arrays of different sizes
  auto C               = nda::vector<int>(mpi_rank + 1, mpi_rank);
  decltype(C) C_gather = nda::lazy_mpi_gather(C, comm, root, true);
  EXPECT_EQ(C_gather.size(), mpi_size * (mpi_size + 1) / 2);
  for (int ofs = 0, r = 0; r < mpi_size; ofs += ++r) {
    auto view = C_gather(nda::range(ofs, ofs + r + 1));
    auto exp  = nda::vector<int>(r + 1, r);
    EXPECT_ARRAY_EQ(exp, view);
  }
}

TEST_F(NDAMpi, ScatterCLayout) {
  // scatter a C-layout array
  auto A_scatter1 = mpi::scatter(A, comm);
  auto rg1        = itertools::chunk_range(0, A.shape()[0], comm.size(), comm.rank());
  auto exp_shape1 = std::array<long, 3>{rg1.second - rg1.first, shape_3d[1], shape_3d[2]};
  EXPECT_EQ(exp_shape1, A_scatter1.shape());
  EXPECT_ARRAY_EQ(A(nda::range(rg1.first, rg1.second), nda::ellipsis{}), A_scatter1);

  // scatter a C-layout array view
  auto A_scatter2 = mpi::scatter(A(0, nda::ellipsis{}), comm);
  auto rg2        = itertools::chunk_range(0, A.shape()[1], comm.size(), comm.rank());
  auto exp_shape2 = std::array<long, 2>{rg2.second - rg2.first, shape_3d[2]};
  EXPECT_EQ(exp_shape2, A_scatter2.shape());
  EXPECT_ARRAY_EQ(A(0, nda::range(rg2.first, rg2.second), nda::ellipsis{}), A_scatter2);
}

TEST_F(NDAMpi, ScatterOtherLayouts) {
  // scatter a non C-layout array by first reshaping it
  constexpr auto perm     = decltype(A2)::layout_t::stride_order;
  constexpr auto inv_perm = nda::permutations::inverse(perm);

  auto A2_scatter   = mpi::scatter(nda::permuted_indices_view<nda::encode(inv_perm)>(A2), comm);
  auto A2_scatter_v = nda::permuted_indices_view<nda::encode(perm)>(A2_scatter);
  auto rg           = itertools::chunk_range(0, A2.shape()[1], comm.size(), comm.rank());
  auto exp_shape    = std::array<long, 3>{shape_3d[0], rg.second - rg.first, shape_3d[2]};
  EXPECT_EQ(exp_shape, A2_scatter_v.shape());
  EXPECT_ARRAY_EQ(A2(nda::range::all, nda::range(rg.first, rg.second), nda::range::all), A2_scatter_v);
}

TEST_F(NDAMpi, LazyScatter) {
  // lazy-scatter a C-layout array
  decltype(A) A_scatter1 = nda::lazy_mpi_scatter(A, comm);
  auto rg1               = itertools::chunk_range(0, A.shape()[0], comm.size(), comm.rank());
  auto exp_shape1        = std::array<long, 3>{rg1.second - rg1.first, shape_3d[1], shape_3d[2]};
  EXPECT_EQ(exp_shape1, A_scatter1.shape());
  EXPECT_ARRAY_EQ(A(nda::range(rg1.first, rg1.second), nda::ellipsis{}), A_scatter1);

  // scatter a C-layout array view
  auto A_scatter2 = nda::array<long, 2>(nda::lazy_mpi_scatter(A(0, nda::ellipsis{}), comm));
  auto rg2        = itertools::chunk_range(0, A.shape()[1], comm.size(), comm.rank());
  auto exp_shape2 = std::array<long, 2>{rg2.second - rg2.first, shape_3d[2]};
  EXPECT_EQ(exp_shape2, A_scatter2.shape());
  EXPECT_ARRAY_EQ(A(0, nda::range(rg2.first, rg2.second), nda::ellipsis{}), A_scatter2);
}

TEST_F(NDAMpi, ReduceCLayout) {
  // reduce an array
  auto A_sum = mpi::reduce(A, comm);
  if (mpi_rank == 0) { EXPECT_ARRAY_EQ(nda::make_regular(A * mpi_size), A_sum); }

  // allreduce an array
  auto A_sum_all = mpi::all_reduce(A, comm);
  EXPECT_ARRAY_EQ(nda::make_regular(A * mpi_size), A_sum_all);

  // allreduce an array view
  auto B     = nda::make_regular(A * (mpi_rank + 1));
  auto B_max = mpi::all_reduce(B(0, 0, _), comm, MPI_MAX);
  EXPECT_ARRAY_EQ(B_max, A(0, 0, _) * mpi_size);
  auto B_min = mpi::reduce(B(0, 0, _), comm, mpi_size - 1, false, MPI_MIN);
  if (mpi_rank == mpi_size - 1) EXPECT_ARRAY_EQ(B_min, A(0, 0, _));
}

TEST_F(NDAMpi, ReduceCLayoutInPlace) {
  // in-place reduce an array
  auto B = A;
  mpi::reduce_in_place(B, comm);
  if (mpi_rank == 0) {
    EXPECT_ARRAY_EQ(nda::make_regular(A * mpi_size), B);
  } else {
    EXPECT_ARRAY_EQ(A, B)
  };

  // in-place allreduce an array
  B = A;
  mpi::all_reduce_in_place(B, comm);
  EXPECT_ARRAY_EQ(nda::make_regular(A * mpi_size), B);

  // in-place (all)reduce an array view
  B = A * (mpi_rank + 1);
  mpi::all_reduce_in_place(B(0, 0, _), comm, MPI_MAX);
  EXPECT_ARRAY_EQ(B(0, 0, _), A(0, 0, _) * mpi_size);
  mpi::reduce_in_place(B(0, 1, _), comm, mpi_size - 1, false, MPI_MIN);
  if (mpi_rank == mpi_size - 1) EXPECT_ARRAY_EQ(B(0, 1, _), A(0, 1, _));
}

TEST_F(NDAMpi, ReduceOtherLayouts) {
  // reduce an array
  auto A2_sum = mpi::reduce(A2, comm);
  if (mpi_rank == 0) { EXPECT_ARRAY_EQ(nda::make_regular(A2 * mpi_size), A2_sum); }

  // allreduce an array
  auto A2_sum_all = mpi::all_reduce(A2, comm);
  EXPECT_ARRAY_EQ(nda::make_regular(A2 * mpi_size), A2_sum_all);

  // all reduce an array view
  decltype(A2) B2 = A2 * (mpi_rank + 1);
  auto B2_max     = mpi::all_reduce(B2(_, 0, 0), comm, MPI_MAX);
  EXPECT_ARRAY_EQ(B2_max, A2(_, 0, 0) * mpi_size);
  auto B2_min = mpi::reduce(B2(_, 0, 0), comm, mpi_size - 1, false, MPI_MIN);
  if (mpi_rank == mpi_size - 1) EXPECT_ARRAY_EQ(B2_min, A2(_, 0, 0));
}

TEST_F(NDAMpi, ReduceOtherLayoutsInPlace) {
  // in-place reduce an array
  auto B2 = A2;
  mpi::reduce_in_place(B2, comm);
  if (mpi_rank == 0) {
    EXPECT_ARRAY_EQ(nda::make_regular(A2 * mpi_size), B2);
  } else {
    EXPECT_ARRAY_EQ(A2, B2)
  };

  // in-place allreduce an array
  B2 = A2;
  mpi::all_reduce_in_place(B2, comm);
  EXPECT_ARRAY_EQ(nda::make_regular(A2 * mpi_size), B2);

  // in-place (all)reduce an array view
  B2 = A2 * (mpi_rank + 1);
  mpi::all_reduce_in_place(B2(_, 0, 0), comm, MPI_MAX);
  EXPECT_ARRAY_EQ(B2(_, 0, 0), A2(_, 0, 0) * mpi_size);
  mpi::reduce_in_place(B2(_, 0, 1), comm, mpi_size - 1, false, MPI_MIN);
  if (mpi_rank == mpi_size - 1) EXPECT_ARRAY_EQ(B2(_, 0, 1), A2(_, 0, 1));
}

TEST_F(NDAMpi, ReduceCustomType) {
  using namespace nda::clef::literals;

  // reduce an array of matrices
  using matrix_t = nda::matrix<double>;
  nda::vector<matrix_t> B(7);
  nda::vector<matrix_t> exp_sum(7);

  for (int i = 0; i < B.extent(0); ++i) {
    B(i) = matrix_t{4, 4};
    B(i)(k_, l_) << i * (mpi_rank + 1) * (k_ + l_);

    exp_sum(i) = matrix_t{4, 4};
    exp_sum(i)(k_, l_) << i * (mpi_size + 1) * mpi_size / 2 * (k_ + l_);
  }

  auto B_sum = mpi::all_reduce(B, comm);

  EXPECT_ARRAY_EQ(B_sum, exp_sum);
}

TEST_F(NDAMpi, LazyReduce) {
  // lazy-reduce an array
  decltype(A) A_sum = nda::lazy_mpi_reduce(A, comm);
  if (mpi_rank == 0) {
    EXPECT_ARRAY_EQ(nda::make_regular(A * mpi_size), A_sum);
  } else {
    EXPECT_EQ(A_sum.size(), 0);
  }

  // lazy-allreduce an array in-place
  auto B2 = A2;
  B2      = nda::lazy_mpi_reduce(B2, comm, root, true);
  EXPECT_ARRAY_EQ(nda::make_regular(A2 * mpi_size), B2);
}

TEST_F(NDAMpi, BroadcastTransposedMatrix) {
  nda::matrix<std::complex<double>> M_t = transpose(M);
  nda::matrix<std::complex<double>> N;
  if (mpi_rank == 0) N = M_t;
  mpi::broadcast(N, comm, 0);
  EXPECT_ARRAY_EQ(M_t, N);
}

TEST_F(NDAMpi, BroadcastTransposedArray) {
  nda::array<long, 3> A_t = transpose(A);
  nda::array<long, 3> B(2, 4, 6);
  if (mpi_rank == 0) B = A_t;
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
  auto rg = itertools::chunk_range(0, 7, mpi_size, mpi_rank);
  EXPECT_ARRAY_EQ(B, A(nda::range(rg.first, rg.second), _));
  EXPECT_ARRAY_NEAR(C, B);

  // gather an array
  B *= -1;
  arr_t D = mpi::gather(B, comm);
  if (mpi_rank == 0) { EXPECT_ARRAY_NEAR(D, -A); }

  // broadcast an array
  mpi::broadcast(D, comm);
  EXPECT_ARRAY_NEAR(D, -A);

  // all gather an array
  D() = 0;
  D   = mpi::all_gather(B, comm);
  EXPECT_ARRAY_NEAR(D, -A);

  // reduce an array
  arr_t R1 = mpi::reduce(A, comm);
  if (mpi_rank == 0) { EXPECT_ARRAY_NEAR(R1, mpi_size * A); }

  // all reduce an array
  arr_t R2 = mpi::all_reduce(A, comm);
  EXPECT_ARRAY_NEAR(R2, mpi_size * A);
}

TEST_F(NDAMpi, PassingTemporaryObjects) {
  auto A         = nda::array<int, 1>{1, 2, 3};
  auto lazy_arr  = mpi::gather(nda::array<int, 1>{1, 2, 3}, comm);
  auto res_arr   = nda::array<int, 1>(lazy_arr);
  auto lazy_view = mpi::gather(A(), comm);
  auto res_view  = nda::array<int, 1>(lazy_view);
  if (comm.rank() == 0) {
    for (long i = 0; i < comm.size(); ++i) {
      EXPECT_ARRAY_EQ(res_arr(nda::range(i * 3, (i + 1) * 3)), A);
      EXPECT_ARRAY_EQ(res_view(nda::range(i * 3, (i + 1) * 3)), A);
    }
  } else {
    EXPECT_TRUE(res_arr.empty());
    EXPECT_TRUE(res_view.empty());
  }
}

MPI_TEST_MAIN
