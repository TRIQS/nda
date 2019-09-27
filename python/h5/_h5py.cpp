#include <pybind11/pybind11.h>
#include <h5/h5.hpp>
#include "h5py_io.hpp"

#include <pybind11/stl.h>

using namespace pybind11::literals;
namespace py = pybind11;

// get all keys
// read/write string attribute

PYBIND11_MODULE(_h5py, m) {

  // File
  py::class_<h5::file> File(m, "File");

  File //
     .def(py::init<std::string const &, char>(), "Constructor", "name"_a, "mode"_a)
     .def_property_readonly("name", &h5::file::name, "Name of the file")
     .def("flush", &h5::file::flush, "Flush the file")
     .def("close", &h5::file::close, "Close the file")
     //
     ;

  // Group
  py::class_<h5::group> Group(m, "Group");

  Group //
     .def(py::init<h5::file>(), "Constructor", "f"_a)
     .def_property_readonly("name", &h5::group::name, "Name of the group")
     .def("open_group", &h5::group::open_group, "Open the subgroup", "key"_a)
     .def("create_group", &h5::group::create_group, "Open the subgroup", "key"_a, "delete_if_exists"_a = true)
     .def("keys", &h5::group::get_all_subgroup_dataset_names, "All the keys")
     .def("has_subgroup", &h5::group::has_subgroup, "", "key"_a)
     .def("has_dataset", &h5::group::has_dataset, "", "key"_a)
     .def("write_attribute", [](h5::group g, std::string const &key, std::string const &val) { h5_write_attribute(g, key, val); },
          "Write an attribute", "key"_a, "val"_a)
     .def("read_attribute", [](h5::group g, std::string const &key) { return h5::h5_read_attribute<std::string>(g, key); }, "Read an attribute",
          "key"_a)
     .def_property_readonly("file", &h5::group::get_file, "The parent file")

     //
     ;

  // the general h5_write function for a PyObject
  m.def(
     "h5_write",
     [](h5::group g, std::string const &name, py::object ob) {
       h5::h5_write_bare(g, name, ob.ptr());
       if (PyErr_Occurred()) throw pybind11::error_already_set();
     },
     "g"_a, "name"_a, "ob"_a);

  // the general h5_read function for a PyObject
  m.def(
     "h5_read",
     [](h5::group g, std::string const &name) -> py::object {
       PyObject *ob = h5::h5_read_bare(g, name);
       if (ob == nullptr)
         throw pybind11::error_already_set();
       else
         return py::reinterpret_steal<py::object>(ob);
     },
     "g"_a, "name"_a);
}
