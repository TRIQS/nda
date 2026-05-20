#include <nda/nda.hpp>
#include <iostream>

int main() {
  nda::clef::placeholder<0> i_;
  nda::clef::placeholder<1> j_;
  auto expr = i_ + j_;
  std::cout << nda::clef::eval(expr, i_ = 1.0, j_ = 2.0) << std::endl;
}
