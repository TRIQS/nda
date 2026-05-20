#include <nda/nda.hpp>
#include <iostream>

int main() {
  nda::clef::placeholder<0> i_;
  nda::clef::placeholder<1> j_;
  auto ex = i_ + j_;
  std::cout << nda::clef::eval(ex, i_ = 1, j_ = 2) << std::endl;
}
