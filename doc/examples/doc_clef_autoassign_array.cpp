#include <nda/nda.hpp>
#include <iostream>

int main() {
  nda::clef::placeholder<0> i_;
  nda::clef::placeholder<1> j_;
  nda::array<int, 2> a(2, 3);
  a(i_, j_) << 10 * i_ + j_;
  std::cout << a << std::endl;
}
