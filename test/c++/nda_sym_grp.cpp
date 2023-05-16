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
  // the 4x4 matrix, initialized such that it respects the symmetries
  using idx_t      = std::array<long, 2>;
  using sym_t      = std::tuple<idx_t, nda::operation>;
  using sym_func_t = std::function<sym_t(idx_t const &)>;

  nda::array<std::complex<double>, 2> A(4, 4);

  auto a  = nda::rand();
  A(0, 0) = a;
  A(2, 0) = a;
  A(0, 3) = a;
  A(2, 3) = a;

  auto b  = nda::rand();
  A(1, 0) = b;
  A(1, 3) = b;

  auto c  = nda::rand();
  A(3, 0) = c;
  A(3, 3) = c;

  auto d  = nda::rand();
  A(0, 1) = d;
  A(2, 1) = d;
  A(0, 2) = d;
  A(2, 2) = d;

  auto e  = nda::rand();
  A(1, 1) = e;
  A(1, 2) = e;

  auto f  = nda::rand();
  A(3, 1) = f;
  A(3, 2) = f;

  // 1) {0, 1, 2, 3} -> {2, 1, 0, 3}
  auto p0 = [](idx_t const &x) {
    auto p   = std::array<long, 4>{2, 1, 0, 3};
    idx_t xp = {p[x[0]], x[1]};
    return sym_t{xp, nda::operation{false, false}};
  };

  // 2) {0, 1, 2, 3} -> {3, 2, 1, 0}
  auto p1 = [](idx_t const &x) {
    auto p   = std::array<long, 4>{3, 2, 1, 0};
    idx_t xp = {x[0], p[x[1]]};
    return sym_t{xp, nda::operation{false, false}};
  };

  // compute symmetry classes
  std::vector<sym_func_t> sym_list = {p0, p1};
  auto grp                         = nda::sym_grp{A, sym_list};

  // test if number of classes matches expectation
  EXPECT_EQ(grp.get_sym_classes().size(), 6);

  // init second array from symmetry group and test if it matches input array
  nda::array<std::complex<double>, 2> B(4, 4);
  auto init_func = [&A](idx_t const &x) { return std::apply(A, x); };
  grp.init(B, init_func);
  EXPECT_EQ_ARRAY(A, B);

  // test symmetrization
  auto const &[max_diff, max_idx] = grp.symmetrize(A);
  EXPECT_EQ(max_diff, 0.0);
  for (auto const x : max_idx) EXPECT_EQ(x, 0);
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
  using idx_t      = std::array<long, 2>;
  using sym_t      = std::tuple<idx_t, nda::operation>;
  using sym_func_t = std::function<sym_t(idx_t const &)>;

  nda::array<std::complex<double>, 2> A(4, 4);

  // 1) A_ij -> A_ji
  auto p0 = [](idx_t const &x) {
    idx_t xp = {x[1], x[0]};
    return sym_t{xp, nda::operation{false, false}};
  };

  // 2) A_ij -> A_(i+1)j
  auto p1 = [](idx_t const &x) {
    idx_t xp = {x[0] + 1, x[1]};
    return sym_t{xp, nda::operation{false, false}};
  };

  // compute symmetry classes
  std::vector<sym_func_t> sym_list = {p0, p1};
  auto grp                         = nda::sym_grp{A, sym_list};

  // test if number of classes matches expectation
  EXPECT_EQ(grp.get_sym_classes().size(), 1);

  // init second array from symmetry group and test if it matches input array
  nda::array<std::complex<double>, 2> B(4, 4);
  auto init_func = [&A](idx_t const &x) { return std::apply(A, x); };
  grp.init(B, init_func);
  EXPECT_EQ_ARRAY(A, B);

  // test symmetrization
  auto const &[max_diff, max_idx] = grp.symmetrize(A);
  EXPECT_EQ(max_diff, 0.0);
  for (auto const x : max_idx) EXPECT_EQ(x, 0);
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
  using idx_t      = std::array<long, 6>;
  using sym_t      = std::tuple<idx_t, nda::operation>;
  using sym_func_t = std::function<sym_t(idx_t const &)>;

  nda::array<std::complex<double>, 6> A(2, 2, 2, 2, 2, 2);

  // 1) A_ijklmn -> A_jkilmn
  auto p0 = [](idx_t const &x) {
    idx_t xp = {x[1], x[2], x[0], x[3], x[4], x[5]};
    return sym_t{xp, nda::operation{false, false}};
  };

  // 2) A_ijklmn -> A_ijkmnl
  auto p1 = [](idx_t const &x) {
    idx_t xp = {x[0], x[1], x[2], x[4], x[5], x[3]};
    return sym_t{xp, nda::operation{false, false}};
  };

  // compute symmetry classes
  std::vector<sym_func_t> sym_list = {p0, p1};
  auto grp                         = nda::sym_grp{A, sym_list};

  // test if number of classes matches expectation
  EXPECT_EQ(grp.get_sym_classes().size(), pow((pow(2, 3) + 2 * 2) / 3, 2));

  // init second array from symmetry group and test if it matches input array
  nda::array<std::complex<double>, 6> B(2, 2, 2, 2, 2, 2);
  auto init_func = [&A](idx_t const &x) { return std::apply(A, x); };
  grp.init(B, init_func);
  EXPECT_EQ_ARRAY(A, B);

  // test symmetrization
  auto const &[max_diff, max_idx] = grp.symmetrize(A);
  EXPECT_EQ(max_diff, 0.0);
  for (auto const x : max_idx) EXPECT_EQ(x, 0);
}