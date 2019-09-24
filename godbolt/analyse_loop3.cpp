
// S2 better than S3
// BUT same if use b.indexmap, c.indexmap
// point is : compute the index ONCE only (compiler can recognize this !).
// OK : analysis is fine, strided linear should be optimal as expected

#include <vector>

#include <nda/nda.hpp>

//[[gnu::noinline]] void ex_tmp(nda::array<double, 3> &a, nda::array<double, 3> &b, nda::array<double, 3> &c) { a = 2 * b + c; }

/*
 void S1(nda::array<double, 3> &a, nda::array<double, 3> &b, nda::array<double, 3> &c,
 int i0 , int i1, int i2) {
 a(i0, i1, i2) = (2 * b + c)(i0, i1, i2);
 }
*/

void S3(nda::array<double, 3> &a, nda::array<double, 3> &b, nda::array<double, 3> &c, int i0, int i1, int i2) {
  a(i0, i1, i2) = 2 * b(i0, i1, i2) + c(i0, i1, i2);
}
/*
void S2(nda::array<double, 3> &a, nda::array<double, 3> &b, nda::array<double, 3> &c,
 int i0 , int i1, int i2) {
     const long st0 = a.indexmap().strides()[0];
  const long st1 = a.indexmap().strides()[1];
  const long st2 = a.indexmap().strides()[2];

a.storage()[i0 * st0 + i1 * st1 + i2] =
      2 * b.storage()[i0 * st0 + i1 * st1 + i2] +
      c.storage()[i0 * st0 + i1 * st1 + i2];

}*/
void S2b(nda::array<double, 3> &a, nda::array<double, 3> &b, nda::array<double, 3> &c, int i0, int i1, int i2) {
  const long st0 = a.indexmap().strides()[0];
  const long st1 = a.indexmap().strides()[1];
  const long st2 = a.indexmap().strides()[2];

  a.storage()[a.indexmap()(i0, i1, i2)] = 2 * b.storage()[a.indexmap()(i0, i1, i2)] + c.storage()[a.indexmap()(i0, i1, i2)];
}

/*
[[gnu::noinline]] void ex_tmp_manual_loop(nda::array<double, 3> &a, nda::array<double, 3> &b, nda::array<double, 3> &c) {
  const long l0 = a.shape()[0];
  const long l1 = a.shape()[0];
  const long l2  = a.shape()[2];
  for (long i0 = 0; i0 < l0; ++i0)
    for (long i1 = 0; i1 < l1; ++i1)
      for (long i2 = 0; i2 < l2; ++i2) { a(i0, i1, i2) = (2 * b + c)(i0, i1, i2); }
}
*/
/*
[[gnu::noinline]] void for_loop(nda::array<double, 3> &a, nda::array<double, 3> &b, nda::array<double, 3> &c) {
  const long l0 = a.shape()[0];
  const long l1 = a.shape()[0];
  const long l2  = a.shape()[2];
  for (long i0 = 0; i0 < l0; ++i0)
    for (long i1 = 0; i1 < l1; ++i1)
      for (long i2 = 0; i2 < l2; ++i2) { a(i0, i1, i2) = 2 * b(i0, i1, i2) + c(i0, i1, i2); }
}
*/
/*

[[gnu::noinline]] void pointers3dloop(nda::array<double, 3> &a, nda::array<double, 3> &b, nda::array<double, 3> &c) {
  const long st0 = a.indexmap().strides()[0];
  const long st1 = a.indexmap().strides()[1];
  const long st2 = a.indexmap().strides()[2];
  const long l0  = a.shape()[0];
  const long l1  = a.shape()[1];
  const long l2  = a.shape()[2];

  double *pb = &(b(0, 0, 0));
  double *pa = &(a(0, 0, 0));
  double *pc = &(c(0, 0, 0));
  for (long i0 = 0; i0 < l0; ++i0)
    for (long i1 = 0; i1 < l1; ++i1)
      for (long i2 = 0; i2 < l2; ++i2) { a.data_start()[i0 * st0 + i1 * st1 + i2*st2] =
      2 * b.data_start()[i0 * st0 + i1 * st1 + i2*st2] +
      c.data_start()[i0 * st0 + i1 * st1 + i2*st2]; }
      //for (long i2 = 0; i2 < l2; ++i2) { pa[i0 * st0 + i1 * st1 + i2] = 2 * pb[i0 * st0 + i1 * st1 + i2] + pc[i0 * st0 + i1 * st1 + i2]; }
}
*/

/*int main() {
     nda::array<double, 3> a, b, c;
     ex_tmp(a,b,c);
}*/
/*
long f1(nda::array<long,1> const & a, long n) { return a(n);}
long f11(nda::array<long,1> const & a, long n) {
    return a.data_start()[n*a.indexmap().strides()[0]];}


long f1(nda::array<long,2> const & a) { return a(0,0);}

*//*
void it1(nda::array<double, 1> & a, nda::array<double, 1> & b,
nda::array<double, 1> & c) {
  const long l0 = a.indexmap().lengths()[0];
    for (long i=0; i <l0; ++i) {
         a(i) = b(i) + c(i);
   }
}

void it2(nda::array<double, 1> & a, nda::array<double, 1> & b
, nda::array<double, 1> & c) {
  const long l0 = a.indexmap().lengths()[0];
    double *pb    = &(b(0));
  double *pa    = &(a(0));
  double *pc    = &(c(0));
    for (long i=0; i <l0; ++i) {
            pa[i] = pb[i] + pc[i];
}

}
*/
