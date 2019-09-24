#include <nda/nda.hpp>

void it2(nda::array<double, 1> &a, nda::array<double, 1> &b, nda::array<double, 1> &c) {
  const long l0 = a.indexmap().lengths()[0];
  double *pb    = &(b(0));
  double *pa    = &(a(0));
  double *pc    = &(c(0));
  for (long i = 0; i < l0; ++i) { pa[i] = pb[i] + pc[i]; }
}
