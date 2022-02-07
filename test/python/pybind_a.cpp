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

//-------------------------------

PYBIND11_MODULE(nda_py_a, m) {

  _import_array();
  //

  py::class_<A> _A(m, "A");
  _A.def(py::init<int>(), "Constructor", "n"_a);

  _A.def("get", &A::get, "doc no const");
  _A.def("get_c", &A::get_c, "doc no const");

  //
}
