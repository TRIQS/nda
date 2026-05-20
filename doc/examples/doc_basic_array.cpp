#include <nda/nda.hpp>
#include <iostream>

int main() {
  // create a regular 3x2 array of ones
  auto arr = nda::ones<int>(3, 2);
  std::cout << arr << std::endl;

  // assign the value 42 to the first row
  arr(0, nda::ellipsis{}) = 42;
  std::cout << arr << std::endl;
}
