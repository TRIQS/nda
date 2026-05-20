#include <nda/nda.hpp>
#include <iostream>
#include <vector>

int main() {
  nda::clef::placeholder<0> i_;
  std::vector<int> v(3);
  nda::clef::make_expr(v)[i_] << 10 * (i_ + 1);
  for (auto x : v) std::cout << x << " ";
  std::cout << std::endl;
}
