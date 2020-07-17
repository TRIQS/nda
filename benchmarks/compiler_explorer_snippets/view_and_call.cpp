// clang 10 :  -std=c++20 -O3 -DNDEBUG -march=skylake
// gcc 9  :    -std=c++17 -O3 -DNDEBUG -march=skylake

#include <nda/nda.hpp>

nda::range_all _;

template <int R>
using Vs = nda::basic_array_view<
   double, R, nda::C_stride_layout, 'A',
   nda::default_accessor, nda::borrowed>;

template <int R>
using V = nda::basic_array_view<
   double, R, nda::C_layout, 'A',
   nda::default_accessor, nda::borrowed>;

double f1(V<4> v, int i, int j, int k, int l) {
  return v(k, _, l, _)(i, j);
}

double f2(V<4> v, int i, int j, int k, int l) {
  return v(k, i, l, j);
}

double g0(V<2> v, int i, int j) {
  return v(i, j);
}
double g0b(Vs<2> v, int i, int j) {
  return v(i, j);
}

double g1(V<2> v, int i, int j) {
  return v(_, _)(i, j);
}

double g2(V<2> v, int i, int j) {
  return v(i, _)(j);
}
