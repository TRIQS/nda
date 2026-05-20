#include <nda/nda.hpp>
#include <iostream>
#include <vector>

int main() {
  nda::clef::placeholder<0> i_;
  auto domain = std::vector{1, 2, 3};
  auto ex     = i_ * i_;
  std::cout << nda::clef::sum(ex, i_ = domain) << std::endl;
}
