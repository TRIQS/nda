#pragma once
#include <pybind11/pybind11.h>

#include <h5/array_interface.hpp>
namespace py = pybind11;

namespace h5 {

  // mae a module in pytriqs/archive, using the wrapping

  // Missing AATRIBUTE

  void h5_write_bare(group g, std::string const &name, py::object ob);

  py::object h5_read_bare(group g, std::string const &name);

} // namespace h5
