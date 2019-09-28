#define NDA_ENFORCE_BOUNDCHECK

#include <pybind11/pybind11.h>
#include <nda/nda_py_interface.hpp>

#include <pybind11/stl.h>

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

// -------------------

namespace pybind11::detail {

  template <typename T, int R>
  struct type_caster<nda::array_view<T, R>> {
    using _type = nda::array_view<T, R>;

    public:
    PYBIND11_TYPE_CASTER(_type, _("nda::array_view"));

    bool load(handle src, bool) {
      PyObject *source = src.ptr();
      if (not nda::python::is_convertible_to_array_view<T, R>(source)) return false;

      nda::python::numpy_proxy p = nda::python::make_numpy_proxy(source);
      value.rebind(nda::python::make_array_view_from_numpy_proxy<T, R>(p));

      if (PyErr_Occurred()) PyErr_Print();
      return true;
    }

    static handle cast(nda::array_view<T, R> src, return_value_policy /* policy */, handle /* parent */) {
      nda::python::numpy_proxy p = nda::python::make_numpy_proxy_from_array(src);
      return p.to_python();
    }
  };

  template <typename T, int R>
  struct type_caster<nda::array<T, R>> {
    using _type = nda::array<T, R>;

    public:
    PYBIND11_TYPE_CASTER(_type, _("nda::array"));

    bool load(handle src, bool) {
      PyObject *source = src.ptr();
      if (not nda::python::is_convertible_to_array_view<T, R>(source)) return false;

      nda::python::numpy_proxy p = nda::python::make_numpy_proxy(source);
      value                      = nda::python::make_array_view_from_numpy_proxy<T, R>(p);

      if (PyErr_Occurred()) PyErr_Print();
      return true;
    }

    static handle cast(nda::array<T, R> src, return_value_policy /* policy */, handle /* parent */) {
      nda::python::numpy_proxy p = nda::python::make_numpy_proxy_from_array(src);
      return p.to_python();
    }
  };
} // namespace pybind11::detail

//-------------------------------

PYBIND11_MODULE(converter_test, m) {

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
