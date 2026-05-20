#include <nda/nda.hpp>
#include <iostream>

int main() {
  // create two arrays A and B
  auto A = nda::array<int, 2>({{1, 2}, {3, 4}});
  auto B = nda::array<int, 2>({{5, 6}, {7, 8}});

  // add them elementwise
  auto ex = A + B; // ex is an nda::expr object

  // evaluate the lazy expression by constructing a new array
  nda::array<int, 2> C = ex;
  std::cout << C << std::endl;

  // evaluate the lazy expression using nda::make_regular
  auto D = nda::make_regular(ex);
  std::cout << D << std::endl;
}
