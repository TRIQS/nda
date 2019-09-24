#include <nda/nda.hpp>
void it1(nda::array<double, 1> &a, nda::array<double, 1> &b, nda::array<double, 1> &c) {
  const long l0 = a.indexmap().lengths()[0];
  for (long i = 0; i < l0; ++i) { a(i) = b(i) + c(i); }
}
