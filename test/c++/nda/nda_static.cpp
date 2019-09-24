#include "./test_common.hpp"

// ==============================================================

TEST(NDA, Create1) { //NOLINT
  nda::array<long, 2> a(3, 3);

  nda::for_each_static<nda::permutations::encode(std::array{3,3})>(a.shape(), [&a](auto x0, auto x1) { a(x0, x1) = x0 + 10*x1; });

  std::cout  << a<<std::endl;

}

	


