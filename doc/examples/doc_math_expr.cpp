#include <nda/nda.hpp>
#include <iostream>

int main() {
  // create an array A
  auto A = nda::array<int, 2>({{1, 2}, {3, 4}});

  // square the elements of A
  auto ex = nda::pow(A, 2); // ex is a lazy expression

  // evaluate the lazy expression by constructing a new array
  nda::array<int, 2> A_sq = ex;
  std::cout << A_sq << std::endl;

  // evaluate the lazy expression using nda::make_regular
  std::cout << nda::make_regular(ex) << std::endl;
}
