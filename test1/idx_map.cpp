#include "gtest.h"
#include <memory>

#define TRIQS_DEBUG_ARRAYS_MEMORY
#define NDA_ENFORCE_BOUNDCHECK

#include <nda/indexmaps/idx_map.hpp>
#include <nda/indexmaps/for_each.hpp>
#include <nda/indexmaps/io.hpp>
#include <triqs/test_tools/arrays.hpp>
//#include <triqs/utility/std_array_addons.hpp>

nda::range _;
nda::ellipsis ___;

using namespace nda;

template <typename... INT> std::array<long, sizeof...(INT)> ma(INT... i) { return {i...}; }

//-----------------------

TEST(idxstat, ConstructStatic) {

  idx_map<3> i1{{1, 2, 3}, layout::C};

  std::cerr << i1 << std::endl;
  EXPECT_TRUE(i1.lengths() == (ma(1, 2, 3)));
  EXPECT_TRUE(i1.strides() == (ma(6, 3, 1)));
}

//-----------------------

TEST(idxstat, eval) {

  idx_map<3> i1{{2, 7, 3}, layout::C};
  EXPECT_TRUE(i1.strides() == (ma(21, 3, 1)));

  EXPECT_EQ(i1(1, 3, 2), 21 * 1 + 3 * 3 + 2 * 1);
}

//-----------------------

TEST(idxstat, boundcheck) {

  idx_map<3> i1{{2, 7, 3}, layout::C};
  //i1(21, 3, 18);
  EXPECT_THROW(i1(21, 3, 18), std::exception);
}

//-----------------------

TEST(idxstat, slice) {

  idx_map<3> i1{{1, 2, 3}, layout::C};

  idx_map<1> i2 = i1(0, _, 2);

  idx_map<1> c2{{2}, {3}, 2, {0}};

  std::cerr << i2 << std::endl;
  std::cerr << c2 << std::endl;
  EXPECT_TRUE(i2 == c2);

  idx_map<3> i3 = i1(_, _, _);
  EXPECT_TRUE(i3 == i1);
}

//-----------------------

TEST(idxstat, ellipsis) {

  idx_map<3> i1{{1, 2, 3}, layout::C};
  idx_map<2> i2 = i1(0, ___);

  idx_map<2> c2{{2, 3}, {3, 1}, 0, {0, 1}};

  std::cerr << i2 << std::endl;
  std::cerr << c2 << std::endl;
  EXPECT_TRUE(i2 == c2);

  idx_map<3> i3 = i1(___);
  EXPECT_TRUE(i3 == i1);
}

//-----------------------

TEST(idxstat, ellipsis2) {

  idx_map<5> i1{{1, 2, 3, 4, 5}, layout::C};
  std::cerr << i1 << std::endl;

  idx_map<2> i2 = i1(0, ___, 3, 2);
  idx_map<2> c2{{2, 3}, {60, 20}, i1(0, 0, 0, 3, 2), {0, 1}};

  std::cerr << i2 << std::endl;
  std::cerr << c2 << std::endl;
  EXPECT_TRUE(i2 == c2);
}

//----------- Iterator ------------

TEST(idxstat, iteratorC) {

  idx_map<3> i1{{2, 3, 4}};

  std::cerr << i1 << std::endl;

  int pos = 0;
  for (auto p : i1) { EXPECT_EQ(p, pos++); }

  pos = 0;
  for (auto [p, i] : enumerate_indices(i1)) {
    EXPECT_EQ(p, pos++);
    std::cerr << i << std::endl;
  }
}

TEST(idxstat, iteratorD) {

  idx_map<3> i1{{1, 2, 3}, layout::Fortran};

  std::cerr << i1 << std::endl;

  int pos = 0;
  for (auto [p, i] : enumerate_indices_in_layout_order(i1)) {
    EXPECT_EQ(p, pos++);
    std::cerr << i << std::endl;
  }
}

TEST(idxstat, for_each) {

  auto l = [](int i, int j, int k) { std::cout << i << " " << j << " " << k << "\n"; };
  for_each(std::array<long,3>{1, 2, 3}, l);
  std::cout  << "-------\n";
  for_each(std::array<long,3>{1, 2, 3}, l, traversal::Fortran);
}

// Different construction
// Cross constrution

MAKE_MAIN;
