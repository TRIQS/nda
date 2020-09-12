// DO NOT EDIT. Generated automatically by cpp2py
#define CPP2PY_GENERATED_PYBIND11_MODULE
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <iostream>
#include <sstream>
#include <algorithm>

#include <vector>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
#include "wrap_basic.hpp"

namespace cpp2py {

  template <typename T>
  T dict_get_default(py::dict kw, const char *name, const char *c_type_str, std::stringstream &fs) {
    try {
      return kw[name].cast<T>();
    } catch (py::error_already_set const &e) { // Strange, we expect index_error but cf pybind11 item_accessor code
      fs << "\n Mandatory parameter '" << name << "' is missing. \n";
    } catch (py::cast_error const &e) {
      fs << "\n The parameter " << name << " does not have the right type : expecting " << c_type_str
         << " in C++, but got '" << Py_TYPE(kw[name].ptr())->tp_name << "' in Python.";
    }
    return {};
  }
} // namespace cpp2py

PYBIND11_MODULE(wrap_basic, MODULE) {

  // ------------ class MemberAccess  ------------------------

  py::class_<member_access> MemberAccess(MODULE, "MemberAccess");

  MemberAccess.def(
     py::init([](py::kwargs kw) -> member_access {
       std::stringstream fs;
       member_access self;
       self.arr =
          (kw.contains("arr") ? cpp2py::dict_get_default<nda::array<long, 1>>(kw, "arr", "nda::array<long, 1>", fs) :
                                nda::array<long, 1>{1, 2, 3});
       self.arr_arr          = (kw.contains("arr_arr") ? cpp2py::dict_get_default<nda::array<nda::array<long, 1>, 1>>(
                          kw, "arr_arr", "nda::array<nda::array<long, 1>, 1>", fs) :
                                                nda::array<nda::array<long, 1>, 1>{{1, 2, 3}, {1, 2}});
       auto all_keys_ordered = std::vector<std::string>{"arr", "arr_arr"};
       for (auto obj : py::reinterpret_steal<py::list>(PyDict_Keys(kw.ptr()))) {
         if (not std::binary_search(all_keys_ordered.begin(), all_keys_ordered.end(), obj.cast<std::string>()))
           fs << "\n The parameter '" << obj.cast<std::string>() << "' is not recognized.";
       }
       if (not fs.str().empty())
         throw std::runtime_error(
            ("\n ---\nError(s) in Python -> C++ constructor of MemberAccess\n " + fs.str()).c_str());
       return self;
     }),
     R"RAW_INNER(
|──────────────────────────────────────────────────────────────────────────|
|   Name    |                 Type                 |      Initializer      |
|──────────────────────────────────────────────────────────────────────────|
|    arr    |         nda::array<long, 1>          |       {1, 2, 3}       |
|──────────────────────────────────────────────────────────────────────────|
|  arr_arr  |  nda::array<nda::array<long, 1>, 1>  |  {{1, 2, 3}, {1, 2}}  |
|──────────────────────────────────────────────────────────────────────────|
)RAW_INNER");
  MemberAccess.def("as_dict", [](member_access const &self) -> py::dict {
    py::dict dic;
    dic["arr"]     = py::cast(self.arr);
    dic["arr_arr"] = py::cast(self.arr_arr);
    return dic;
  });
  MemberAccess.def("__repr__", [](py::handle self) {
    return "MemberAccess (**"
       + py::reinterpret_steal<py::object>(PyObject_Repr(self.attr("as_dict")().ptr())).cast<std::string>() + ')';
  });

  // ------------ methods MemberAccess-------------

  MemberAccess.def_readwrite("arr", &member_access::arr, "");
  MemberAccess.def_readwrite("arr_arr", &member_access::arr_arr, "");

  // ------------ end class MemberAccess-------------

  MODULE.def("make_arr", py::overload_cast<long>(&make_arr), "", py::arg("n"));
  MODULE.def("make_arr", py::overload_cast<long, long>(&make_arr), "", py::arg("n1"), py::arg("n2"));
  MODULE.def("make_arr_arr", py::overload_cast<long, long>(&make_arr_arr), "", py::arg("n1"), py::arg("n2"));
  MODULE.def("size_arr", py::overload_cast<nda::array<long, 1> const &>(&size_arr), "", py::arg("a"));
  MODULE.def("size_arr", py::overload_cast<nda::array<long, 2> const &>(&size_arr), "", py::arg("a"));
  MODULE.def("size_arr_v", py::overload_cast<nda::array_view<long, 1>>(&size_arr_v), "", py::arg("a"));
  MODULE.def("size_arr_v", py::overload_cast<nda::array_view<long, 2>>(&size_arr_v), "", py::arg("a"));
  MODULE.def("size_arr_cv", py::overload_cast<nda::array_const_view<long, 1>>(&size_arr_cv), "", py::arg("a"));
  MODULE.def("size_arr_cv", py::overload_cast<nda::array_const_view<long, 2>>(&size_arr_cv), "", py::arg("a"));
  MODULE.def("size_arr_arr", py::overload_cast<nda::array<nda::array<long, 1>, 1>>(&size_arr_arr), "", py::arg("a"));
  MODULE.def("size_arr_arr_v", py::overload_cast<nda::array<nda::array_view<long, 1>, 1>>(&size_arr_arr_v), "",
             py::arg("a"));
  MODULE.def("size_arr_arr_cv", py::overload_cast<nda::array<nda::array_const_view<long, 1>, 1>>(&size_arr_arr_cv), "",
             py::arg("a"));
  MODULE.def("multby2", &multby2<1>, "", py::arg("a"));
  MODULE.def("multby2", &multby2<2>, "", py::arg("a"));
  MODULE.def("multby2_d", py::overload_cast<nda::array<double, 1> const &>(&multby2_d), "", py::arg("a"));

  //  ------------ Initialization of the module if any ------------
}