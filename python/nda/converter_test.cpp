#include <pybind11/pybind11.h>
#include <nda/nda_py_interface.hpp>

#include <pybind11/stl.h>

using namespace pybind11::literals;
namespace py = pybind11;

double f(nda::array_view<double, 2> a) { return a(1, 2); }

namespace pybind11::detail {

  template <typename T, int R>
  struct type_caster<nda::array_view<T, R>> {
    using _type = nda::array_view<T, R>;

    public:
    /**
         * This macro establishes the name 'inty' in
         * function signatures and declares a local variable
         * 'value' of type inty
         */
    PYBIND11_TYPE_CASTER(_type, _("array_view"));

    /**
         * Conversion part 1 (Python->C++): convert a PyObject into a inty
         * instance or return false upon failure. The second argument
         * indicates whether implicit conversions should be applied.
         */
    bool load(handle src, bool) {
      /* Extract PyObject from handle */
      PyObject *source = src.ptr();
      /* Try converting into a Python integer value */

      nda::python::numpy_rank_and_type rt = nda::python::get_numpy_rank_and_type(source);
      nda::python::view_info vi           = nda::python::from_python(rt);
      value                               = nda::python::make_array_view_from_view_info<T, R>(vi);
      return true; // CONTROL ERROR :

      //return !(value.long_value == -1 && !PyErr_Occurred());
    }

    /**
     * Conversion part 2 (C++ -> Python): convert an inty instance into
     * a Python object. The second and third arguments are used to
     * indicate the return value policy and parent object (for
     * ``return_value_policy::reference_internal``) and are generally
     * ignored by implicit casters.
     */
    static handle cast(nda::array_view<T, R> src, return_value_policy /* policy */, handle /* parent */) {
      nda::python::view_info vi = nda::python::make_view_info_from_array(src);
      return nda::python::to_python(vi);
    }
  };
} // namespace pybind11::detail

PYBIND11_MODULE(converter_test, m) {

  _import_array();
  //

  m.def("f", &f, "DOC");

  //
}
