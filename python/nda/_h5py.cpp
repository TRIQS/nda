#include <pybind11/pybind11.h>

#include <h5/h5.hpp>
#include "python.hpp"

using namespace pybind11::literals;
namespace py = pybind11;

// get all keys
// read/write string attribute

PYBIND11_MODULE(_h5py, m) {

  // File
  py::class_<h5::file> File(m, "File");

  File //
     .def(py::init<std::string const &, char>(), "Constructor", "name"_a, "mode"_a)
     .def_property_readonly("name", &h5::file::name, "Name of the file");

 
  // Group
  py::class_<h5::group> Group(m, "Group");

  Group //
     .def(py::init<h5::file>(), "Constructor", "f"_a)
     .def_property_readonly("name", &h5::group::name, "Name of the group")
     .def("open_group", &h5::group::open_group, "Open the subgroup", "key"_a)
     .def("create_group", &h5::group::create_group, "Open the subgroup", "key"_a, "delete_if_exists"_a = false)

     //
     ;

  m.def("h5_write",
        [](h5::group g, std::string const &name, py::object ob) {
          h5::h5_write_bare(g, name, ob.ptr());
          if (PyErr_Occurred()) throw pybind11::error_already_set();
        },
        "g"_a, "name"_a, "ob"_a);

  m.def("h5_read",
        [](h5::group g, std::string const &name) -> py::object {
          PyObject *ob = h5::h5_read_bare(g, name);
          if (ob == nullptr)
            throw pybind11::error_already_set();
          else
            return py::reinterpret_steal<py::object>(ob);
        },
        "g"_a, "name"_a);
}
