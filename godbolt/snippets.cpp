
long f1(nda::array<long, 1> const &a, long n) { return a(n); }
long f11(nda::array<long, 1> const &a, long n) { return a.data_start()[n * a.indexmap().strides()[0]]; }

long f1(nda::array<long, 2> const &a) { return a(0, 0); }

void it1(nda::array<double, 1> const &a) {
  const long l0 = a.indexmap().lengths()[0];
  for (long i = 0; i < l0; ++i) {
    a.storage()[i] = 10 * i;
    //*(a.data_start() + i) = 10*i;
  }
}

void it2(nda::array<double, 1> &a) {
  const long l0 = a.indexmap().lengths()[0];
  for (auto it = a.begin(); it != a.end(); ++it) { *it = 10 * it.indices()[0]; }
}

/*

void f(int i, nda::array<double, 5> & a, double x) { 
 nda::range_all _;
  a(i,i,i, _, _)(0,0) = x;
}


void g(int i, nda::array<double, 5> & a, double x) { 
  a(i,i,i,0,0) = x;
}


*/
// ------- ok
// offset difference with direct call : TO BE RESTESTED

void g1(int i, nda::array<double, 2> &a, double x) {
  nda::range_all _;
  a(i, _)(0) = x;
}

void g2(int i, nda::array<double, 2> &a, double x) { a(i, 0) = x; }

// ---------------------------- Expr template

void e1(nda::array<double, 1> &a, nda::array<double, 1> const &b, nda::array<double, 1> const &c) { a = b + c; }

void e1a(nda::array_view<double, 1> &a, nda::array<double, 1> const &b, nda::array<double, 1> const &c) {
  const long l0 = a.indexmap().lengths()[0];
  for (long i = 0; i < l0; ++i) a(i) = b(i) + c(i);
}

void e1M(nda::array<double, 1> &a, nda::array<double, 1> const &b, nda::array<double, 1> const &c) {
  const long l0 = a.indexmap().lengths()[0];
  double *pa    = &(a(0));
  double *pb    = &(b(0));
  double *pc    = &(c(0));

  for (long i = 0; i < l0; ++i) pa[i] = pb[i] + pc[i];
}


// -----------------------------------------

void g1(int i, nda::array<double, 4> &a, double x) {
  nda::range_all _;
  a(i, i, _, _)(0, 0) = x;
}

void g2(int i, nda::array<double, 4> &a, double x) { a(i, i, 0, 0) = x; }

nda::range_all _;
a(i, i, i, _, _)(0, 0) = x;
}

void g(int i, nda::array<double, 5> &a, double x) { a(i, i, i, 0, 0) = x; }

// =----------------------


void f1(int i, nda::array<double, 4> & a, double x) { 
 nda::range_all _; 
  a(i, _,i, _)(0,0) = x;
}


void f2(int i, nda::array<double, 4> & a, double x) { 
  a(i,0,i,0) = x;
} 

