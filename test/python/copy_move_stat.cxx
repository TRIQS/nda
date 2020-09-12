// DO NOT EDIT. Generated automatically by cpp2py
#define CPP2PY_GENERATED_PYBIND11_MODULE
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <iostream>
#include <sstream>
#include <algorithm>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
#include "copy_move_stat.hpp"

namespace cpp2py {}

PYBIND11_MODULE(copy_move_stat, MODULE) {

  // ------------ class CopyMoveStat  ------------------------

  py::class_<copy_move_stat> CopyMoveStat(MODULE, "CopyMoveStat");

  CopyMoveStat.def(py::init<bool>(), "", py::arg("verbose") = true);

  // ------------ methods CopyMoveStat-------------

  CopyMoveStat.def_static("copy_count", py::overload_cast<>(&copy_move_stat::copy_count), "");
  CopyMoveStat.def_static("move_count", py::overload_cast<>(&copy_move_stat::move_count), "");
  CopyMoveStat.def_static("reset", py::overload_cast<>(&copy_move_stat::reset), "");

  // ------------ end class CopyMoveStat-------------

  // ------------ class MemberStat  ------------------------

  py::class_<member_stat> MemberStat(MODULE, "MemberStat");

  MemberStat.def(py::init<>(), "");
  MemberStat.def_readwrite("m", &member_stat::m, "");

  // ------------ end class MemberStat-------------

  MODULE.def("make_obj", py::overload_cast<>(&make_obj), "");
  MODULE.def("make_arr", py::overload_cast<long>(&make_arr), "", py::arg("n"));
  MODULE.def("make_arr", py::overload_cast<long, long>(&make_arr), "", py::arg("n1"), py::arg("n2"));
  MODULE.def("take_obj", py::overload_cast<copy_move_stat>(&take_obj), "", py::arg("o"));
  MODULE.def("take_arr", py::overload_cast<nda::array<copy_move_stat, 1> const &>(&take_arr), "", py::arg("a"));
  MODULE.def("take_arr", py::overload_cast<nda::array<copy_move_stat, 2> const &>(&take_arr), "", py::arg("a"));

  //  ------------ Initialization of the module if any ------------
}