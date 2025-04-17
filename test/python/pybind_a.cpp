// Copyright (c) 2019--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

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
