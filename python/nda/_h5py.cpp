#include <pybind11/pybind11.h>

#include <h5/h5.hpp>
#include "python.hpp"

using namespace pybind11::literals;
namespace py = pybind11;

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


  //m.def("h5_write", &h5::h5_write, "name"_a, "ob"_a);
  m.def("h5_read", py::overload_cast<h5::group, std::string const &>(h5::h5_read_bare), "g"_a, "name"_a);
  m.def("h5_write", py::overload_cast<h5::group, std::string const &, py::object>(h5::h5_write_bare));


}
