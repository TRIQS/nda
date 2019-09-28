#define NDA_ENFORCE_BOUNDCHECK

#include <pybind11/pybind11.h>
#include <nda_py/nda_py.hpp>
#include <pybind11/stl.h>

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
