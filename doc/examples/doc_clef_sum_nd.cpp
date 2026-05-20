#include <nda/nda.hpp>
#include <iostream>
#include <vector>

int main() {
  nda::clef::placeholder<0> i_;
  nda::clef::placeholder<0> j_;
  auto domain1 = std::vector{1, 2, 3};
  auto domain2 = std::vector{4, 5, 6};
  auto ex      = i_ + j_;
  std::cout << nda::clef::sum(ex, i_ = domain1, j_ = domain2) << std::endl;
}
