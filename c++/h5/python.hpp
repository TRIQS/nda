#pragma once

#include "./array_interface.hpp"

namespace h5 {

  // mae a module in pytriqs/archive, using the wrapping

  // Missing AATRIBUTE

  void h5_write(group g, std::string const &name, PyObject *ob);

  PyObject * h5_read(group g, std::string const &name);

} // namespace h5
