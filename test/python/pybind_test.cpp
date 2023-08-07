// Copyright (c) 2019-2020 Simons Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Authors: Olivier Parcollet, Nils Wentzell

#define NDA_ENFORCE_BOUNDCHECK

#include <pybind11/pybind11.h>
#include <nda_py/nda_py.hpp>
#include <nda_py/pybind11_converters.hpp>
#include "a.hpp"

using namespace pybind11::literals;
namespace py = pybind11;

struct BigObject {
  int i;

  BigObject(BigObject &&)                 = default;
  BigObject(BigObject const &)            = delete;
  BigObject &operator=(BigObject &&)      = default;
  BigObject &operator=(BigObject const &) = delete;
};

namespace pybind11::detail {

  template <>
  struct type_caster<BigObject> {

    public:
    PYBIND11_TYPE_CASTER(BigObject, _("BigObject"));

    bool load(handle src, bool) { return true; }

    static handle cast(BigObject src, return_value_policy /* policy */, handle /* parent */) { return PyLong_FromLong(src.i); }
  };

} // namespace pybind11::detail

BigObject maker() { return {}; }

double f(nda::array_view<double, 2> a, int i, int j) { return a(i, j); }
//double f2(nda::array_view<double, 2> a, int i, int j) { return a(i, j); }
//double f3(nda::array_view<double, 2> a, int i, int j) { return a(i, j); }
//double f4(nda::array_view<double, 2> a, int i, int j) { return a(i, j); }
//double f5(nda::array_view<double, 2> a, int i, int j) { return a(i, j); }
//double f6(nda::array_view<double, 2> a, int i, int j) { return a(i, j); }
//double f7(nda::array_view<double, 2> a, int i, int j) { return a(i, j); }
//double f8(nda::array_view<double, 2> a, int i, int j) { return a(i, j); }
//double f9(nda::array_view<double, 2> a, int i, int j) { return a(i, j); }
//double f10(nda::array_view<double, 2> a, int i, int j) { return a(i, j); }
//double f11(nda::array_view<double, 2> a, int i, int j) { return a(i, j); }
//double f12(nda::array_view<double, 2> a, int i, int j) { return a(i, j); }

//-------------------------------

PYBIND11_MODULE(nda_py_converter_test, m) {

  _import_array();
  //
  pybind11::module::import("nda_py_a");

  m.def("maker", &maker, "dd");
  m.def("f", &f, "DOC");
  //m.def("f2", &f2, "DOC");
  //m.def("f3", &f3, "DOC");
  //m.def("f4", &f4, "DOC");
  //m.def("f5", &f5, "DOC");
  //m.def("f6", &f6, "DOC");
  //m.def("f7", &f7, "DOC");
  //m.def("f8", &f8, "DOC");
  //m.def("f9", &f9, "DOC");
  //m.def("f10", &f10, "DOC");
  //m.def("f11", &f11, "DOC");
  //m.def("f12", &f12, "DOC");

  m.def("ma", &ma, "DOC");

  m.def(
     "make_A", []() { return A{5}; }, "DOC");

  //py::class_<A> _A(m, "A");
  //_A.def(py::init<int>(), "Constructor", "n"_a);

  //_A.def("get", &A::get, "doc no const");
  //_A.def("get_c", &A::get_c, "doc no const");

  //
}
