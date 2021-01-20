// DO NOT EDIT. Generated automatically by cpp2py
#define CPP2PY_GENERATED_PYBIND11_MODULE
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <iostream>
#include <sstream>
#include <algorithm>

#include <pybind11/pybind11.h>
namespace py = pybind11;

#include "wrap_basic.hpp"

namespace cpp2py {}

PYBIND11_MODULE(wrap_basic, MODULE) {

  // ------------ class MemberAccess  ------------------------

  py::class_<member_access> MemberAccess(MODULE, "MemberAccess");

  // ------------ methods MemberAccess-------------

  MemberAccess.def_readwrite("arr", &member_access::arr, "");
  MemberAccess.def_readwrite("arr_arr", &member_access::arr_arr, "");

  // ------------ end class MemberAccess-------------

  MODULE.def("make_arr", py::overload_cast<long>(&make_arr), "", py::arg("n"));
  MODULE.def("make_arr", py::overload_cast<long, long>(&make_arr), "", py::arg("n1"), py::arg("n2"));
  MODULE.def("make_arr_arr", py::overload_cast<long, long>(&make_arr_arr), "", py::arg("n1"), py::arg("n2"));
  MODULE.def("size_arr", py::overload_cast<const nda::array<long, 1> &>(&size_arr), "", py::arg("a"));
  MODULE.def("size_arr", py::overload_cast<const nda::array<long, 2> &>(&size_arr), "", py::arg("a"));
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
  MODULE.def("multby2_d", py::overload_cast<const nda::array<double, 1> &>(&multby2_d), "", py::arg("a"));

  //  ------------ Initialization of the module if any ------------
}
