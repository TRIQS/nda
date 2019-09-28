#define NDA_ENFORCE_BOUNDCHECK

#include <pybind11/pybind11.h>
#include <nda_py/nda_py.hpp>
#include <nda_py/pybind11_converters.hpp>

using namespace pybind11::literals;
namespace py = pybind11;

double f(nda::array_view<double, 2> a, int i, int j) { return a(i, j); }

nda::array<long, 1> ma(int n) {
  nda::array<long, 1> result(n);
  for (int i = 0; i < n; ++i) result(i) = i + 1;
  return result;
}

struct A {
  nda::array<long, 1> a;
  A(int n) { a = ma(n); }
  nda::array_view<long, 1> get() { return a; }
  nda::array_view<long const, 1> get_c() const { return a; }
};

//-------------------------------

PYBIND11_MODULE(nda_py_converter_test, m) {

  _import_array();
  //

  m.def("f", &f, "DOC");
  m.def("ma", &ma, "DOC");

  py::class_<A> _A(m, "A");
  _A.def(py::init<int>(), "Constructor", "n"_a);

  _A.def("get", &A::get, "doc no const");
  _A.def("get_c", &A::get_c, "doc no const");

  //
}
