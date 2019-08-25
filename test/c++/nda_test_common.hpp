#include <gtest/gtest.h> // NOLINT
#include <cmath>
#include <limits>

#define NDA_ENFORCE_BOUNDCHECK
#include <nda/array.hpp>
#include <nda/gtest_tools.hpp>

template <int R> using myshape_t = nda::shape_t<R>;

// variables for the test
nda::range _;
nda::ellipsis ___;

#define MAKE_MAIN_MPI                                                                                                                                \
  int main(int argc, char **argv) {                                                                                                                  \
    ::mpi::environment env(argc, argv);                                                                                                              \
    ::testing::InitGoogleTest(&argc, argv);                                                                                                          \
    return RUN_ALL_TESTS();                                                                                                                          \
  }

#define MAKE_MAIN                                                                                                                                    \
  int main(int argc, char **argv) {                                                                                                                  \
    ::testing::InitGoogleTest(&argc, argv);                                                                                                          \
    return RUN_ALL_TESTS();                                                                                                                          \
  }

