#include <nda/nda.hpp>
#include <iostream>

int main() {
  // create a regular 3x3 array of ones
  auto arr = nda::ones<int>(3, 3);
  std::cout << arr << std::endl;

  // zero out the first column
  arr(nda::range::all, 0) = 0;
  std::cout << arr << std::endl;
}
