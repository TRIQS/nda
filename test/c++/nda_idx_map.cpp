#define NDA_ENFORCE_BOUNDCHECK
#include <gtest/gtest.h> // NOLINT

#include <nda.hpp>

nda::range _;
nda::ellipsis ___;

using namespace nda;

template <typename... INT> std::array<long, sizeof...(INT)> ma(INT... i) { return {i...}; }

//-----------------------

TEST(idxstat, Construct) { // NOLINT

  idx_map<3> i1{{1, 2, 3}};

  std::cerr << i1 << std::endl;
  EXPECT_TRUE(i1.lengths() == (ma(1, 2, 3))); //NOLINT
  EXPECT_TRUE(i1.strides() == (ma(6, 3, 1))); //NOLINT
}

//-----------------------

TEST(idxstat, eval) { // NOLINT

  idx_map<3> i1{{2, 7, 3}};
  EXPECT_TRUE(i1.strides() == (ma(21, 3, 1))); //NOLINT

  EXPECT_EQ(i1(1, 3, 2), 21 * 1 + 3 * 3 + 2 * 1); //NOLINT
}

//-----------------------

TEST(idxstat, boundcheck) { // NOLINT

  idx_map<3> i1{{2, 7, 3}};
  //i1(21, 3, 18);
  EXPECT_THROW(i1(21, 3, 18), std::exception); //NOLINT
}

//-----------------------

TEST(idxstat, slice) { // NOLINT

  idx_map<3> i1{{1, 2, 3}};

  idx_map<1> i2 = i1(0, _, 2);

  idx_map<1> c2{{2}, {3}, 2};

  std::cerr << i2 << std::endl;
  std::cerr << c2 << std::endl;
  EXPECT_TRUE(i2 == c2); //NOLINT

  idx_map<3> i3 = i1(_, _, _);
  EXPECT_TRUE(i3 == i1); //NOLINT
}

//-----------------------

TEST(idxstat, ellipsis) { // NOLINT

  idx_map<3> i1{{1, 2, 3}};
  idx_map<2> i2 = i1(0, ___);

  idx_map<2> c2{{2, 3}, {3, 1}, 0};

  std::cerr << i2 << std::endl;
  std::cerr << c2 << std::endl;
  EXPECT_TRUE(i2 == c2); //NOLINT

  idx_map<3> i3 = i1(___);
  EXPECT_TRUE(i3 == i1); //NOLINT
}

//-----------------------

TEST(idxstat, ellipsis2) { // NOLINT

  idx_map<5> i1{{1, 2, 3, 4, 5}};
  std::cerr << i1 << std::endl;

  idx_map<2> i2 = i1(0, ___, 3, 2);
  idx_map<2> c2{{2, 3}, {60, 20}, i1(0, 0, 0, 3, 2)};

  std::cerr << i2 << std::endl;
  std::cerr << c2 << std::endl;
  EXPECT_TRUE(i2 == c2); //NOLINT
}

//----------- Iterator ------------

TEST(idxstat, iteratorC) { // NOLINT

  idx_map<3> i1{{2, 3, 4}};

  std::cerr << i1 << std::endl;

  int pos = 0;
  for (auto p : i1) {
    EXPECT_EQ(p, pos++); //NOLINT
  }

  pos = 0;
  //for (auto [p, i] : enumerate_indices(i1)) {
    //EXPECT_EQ(p, pos++); //NOLINT
    //std::cerr << i << std::endl;
  //}
}
/*
TEST(idxstat, iteratorD) { // NOLINT

  idx_map<3> i1{{1, 2, 3}, layout::Fortran};

  std::cerr << i1 << std::endl;

  int pos = 0;
  for (auto [p, i] : enumerate_indices_in_layout_order(i1)) {
    EXPECT_EQ(p, pos++);//NOLINT
    std::cerr << i << std::endl;
  }
}
*/
TEST(idxstat, for_each) { // NOLINT

  {
    std::stringstream fs;
    auto l = [&fs](int i, int j, int k) { fs << i << j << k << " "; };

    for_each(std::array<long, 3>{1, 2, 3}, l);
    EXPECT_EQ(fs.str(), "000 001 002 010 011 012 ");
  }

  //{
    //std::stringstream fs;
    //auto l = [&fs](int i, int j, int k) { fs << i << j << k << " "; };

    //for_each(std::array<long, 3>{1, 2, 3}, l, traversal::Fortran);
    //EXPECT_EQ(fs.str(), "000 010 001 011 002 012 ");
  //}
}

// Different construction
// Cross constrution

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
