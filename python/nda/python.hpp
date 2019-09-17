#pragma once
#include <h5/array_interface.hpp>
#include <Python.h>

namespace h5 {

  void h5_write_bare(group g, std::string const &name, PyObject *ob);
  PyObject *h5_read_bare(group g, std::string const &name);

} // namespace h5
