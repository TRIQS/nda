#include <triqs/test_tools/arrays.hpp>
#include <triqs/arrays.hpp>
using namespace triqs::arrays;

//using dcomplex = std::complex<double>;

#define EXPECT_SHAPE1(X, s1) EXPECT_EQ(X.shape(), (mini_vector<size_t, 1>{s1});
#define EXPECT_SHAPE2(X, s1, s2) EXPECT_EQ(X.shape(), (mini_vector<size_t, 2>{s1, s2});
#define EXPECT_SHAPE3(X, s1, s2, s3) EXPECT_EQ(X.shape(), (mini_vector<size_t, 3>{s1, s2, s3});


