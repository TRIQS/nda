#include <nda/nda.hpp>
#include <iostream>

int main() {
  nda::clef::placeholder<0> i_;
  nda::clef::placeholder<1> j_;
  auto ex = i_ + j_;
  auto f  = nda::clef::make_function(ex, i_, j_);
  std::cout << f(1, 2) << std::endl;
  std::cout << f(1.5, 2) << std::endl;
}
