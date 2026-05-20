#include <nda/nda.hpp>
#include <iostream>

int main() {
  // multiply two arrays elementwise
  auto A1               = nda::array<int, 2>({{1, 2}, {3, 4}});
  auto A2               = nda::array<int, 2>({{5, 6}, {7, 8}});
  nda::array<int, 2> A3 = A1 * A2;
  std::cout << A3 << std::endl;

  // multiply two matrices
  auto M1             = nda::matrix<int>({{1, 2}, {3, 4}});
  auto M2             = nda::matrix<int>({{5, 6}, {7, 8}});
  nda::matrix<int> M3 = M1 * M2;
  std::cout << M3 << std::endl;
}
