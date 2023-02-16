#include "./test_common.hpp"
#include <nda/sym_grp.hpp>

// -------------------------------------
/*
    Consider the permutations: 1) {0, 1, 2, 3} -> {2, 1, 0, 3} and 
    2) {0, 1, 2, 3} -> {3, 2, 1, 0}. We assume 4x4 matrix A is invariant 
    when 1) is applied as a row and 2) as a column index transformation. 
    Thus, there should be 6 groups (a to f) of equivalent elements in A:

                    0   1   2   3
                0   a   d   d   a
                1   b   e   e   b
                2   a   d   d   a
                3   c   f   f   c
*/

TEST(SymGrp, MatrixPermutation) { //NOLINT
  // the 4x4 matrix
  nda::array<std::complex<double>, 2> A(4, 4);
 
  // 1) {0, 1, 2, 3} -> {2, 1, 0, 3}
  std::function<nda::operation(std::array<long, 2> &)> p0 = [](std::array<long, 2> &x) {
    auto p = std::array<long, 4>{2, 1, 0, 3};
    x[0]   = p[x[0]];
    return nda::operation{false, false};
  };

  // 2) {0, 1, 2, 3} -> {3, 2, 1, 0}
  std::function<nda::operation(std::array<long, 2> &)> p1 = [](std::array<long, 2> &x) {
    auto p = std::array<long, 4>{3, 2, 1, 0};
    x[1]   = p[x[1]];
    return nda::operation{false, false};
  };

  // compute symmetry classes
  std::vector<std::function<nda::operation(std::array<long, 2> &)>> sym_list = {p0, p1};
  nda::sym_grp grp(A, sym_list);
  EXPECT_EQ(grp.get_sym_classes().size(), 6);
}

// -------------------------------------

// -------------------------------------
/*
    Consider the transformations 1) A_ij -> A_ji and 2) A_ij -> A_(i+1)j.
    If matrix A is supposed to be invariant, then all its elements should fall 
    into one symmetry class.
*/

TEST(SymGrp, MatrixFlipShift) { //NOLINT
  // the 4x4 matrix
  nda::array<std::complex<double>, 2> A(4, 4);
 
  // 1) A_ij -> A_ji
  std::function<nda::operation(std::array<long, 2> &)> p0 = [](std::array<long, 2> &x) {
    auto idx = x[0];
    x[0]     = x[1];
    x[1]     = idx;
    return nda::operation{false, false};
  };

  // 2) A_ij -> A_(i+1)j
  std::function<nda::operation(std::array<long, 2> &)> p1 = [](std::array<long, 2> &x) {
    ++x[0];
    return nda::operation{false, false};
  };

  // compute symmetry classes
  std::vector<std::function<nda::operation(std::array<long, 2> &)>> sym_list = {p0, p1};
  nda::sym_grp grp(A, sym_list);
  EXPECT_EQ(grp.get_sym_classes().size(), 1);
}

// -------------------------------------

// -------------------------------------
/*
    Consider the transformations 1) A_ijklmn -> A_jkilmn and
    2) A_ijklmn -> A_ijkmnl for a rank 6 tensor A. With N elements per dimension
    this should result in ((N^3 + 2N) / 3)^2 symmetry classes.
*/

TEST(SymGrp, TensorCylicTriplet) { //NOLINT
  // the rank 6 tensor
  int N = 2;
  nda::array<std::complex<double>, 6> A(2, 2, 2, 2, 2, 2);
 
  // 1) A_ijklmn -> A_jkilmn
  std::function<nda::operation(std::array<long, 6> &)> p0 = [](std::array<long, 6> &x) {
    auto idx = x[0];
    x[0]     = x[1];
    x[1]     = x[2];
    x[2]     = idx;
    return nda::operation{false, false};
  };

  // 2) A_ijklmn -> A_ijkmnl
  std::function<nda::operation(std::array<long, 6> &)> p1 = [](std::array<long, 6> &x) {
    auto idx = x[3];
    x[3]     = x[4];
    x[4]     = x[5];
    x[5]     = idx;
    return nda::operation{false, false};
  };

  // compute symmetry classes
  std::vector<std::function<nda::operation(std::array<long, 6> &)>> sym_list = {p0, p1};
  nda::sym_grp grp(A, sym_list);
  EXPECT_EQ(grp.get_sym_classes().size(), pow((pow(N, 3) + 2 * N) / 3, 2));
}

// -------------------------------------