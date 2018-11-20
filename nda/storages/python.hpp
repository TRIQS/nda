/*******************************************************************************
 *
 * TRIQS: a Toolbox for Research in Interacting Quantum Systems
 *
 * Copyright (C) 2011-2017 by O. Parcollet
 * Copyright (C) 2018 by Simons Foundation
 *   author : O. Parcollet
 *
 * TRIQS is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * TRIQS is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * TRIQS. If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/
#pragma once
#include "./python.hpp"
#include <cpp2py/pyref.hpp>
#include <triqs/utility/exceptions.hpp>

// SHOULD ONLY BE INCLUDED in a python module.

#ifndef PYTHON_NUMPY_VERSION_LT_17
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif

#include "Python.h"
#include <numpy/arrayobject.h>

namespace nda {

  // ----------------  Utilities -------------------

  static void py_decref(PyObject *x) { PY_DECREF(x); }

  template <typename T> static void delete_pycapsule(PyObject *capsule) {
    handle<T, 'S'> *handle = static_cast<handle<T, 'S'> *>(PyCapsule_GetPointer(capsule, "guard"));
    handle->decref();
    delete handle;
  }
  // -------------  make_handle ------------

  // Take a handle on a numpy. numpy is a borrowed Python ref.
  // implemented only in Python module, not in triqs cpp
  template <typename T> handle<T, 'S'> make_handle(PyObject *ob) {

    _import_array();

    if (obj == NULL) TRIQS_RUNTIME_ERROR << " Can not build an mem_blk_handle from a NULL PyObject *";
    Py_INCREF(obj); // assume borrowed
    if (!PyArray_Check(obj)) TRIQS_RUNTIME_ERROR << "Internal error : ref_counter construct from pyo : obj is not an array";
    PyArrayObject *arr = (PyArrayObject *)(obj);

    handle<'S'> r;
    r.data        = (T *)PyArray_DATA(arr);
    r.size        = size_t(PyArray_SIZE(arr));
    r.id          = r.rtable.get();
    r.sptr        = ob;
    r.release_fnt = &py_decref;
    return r;
  }

  // ------------------  make_pycapsule  ----------------------------------------------------
  // make a pycapsule out of the shared handle to return to Python

  template <typename T> PyObject *make_pycapsule(handle<T, 'S'> const &h) {
    h.incref();
    void *keep = new handle<T, 'S'>{h};
    return PyCapsule_New(keep, "guard", &delete_pycapsule<T>);
  }

  template <typename T> PyObject *make_pycapsule(handle<T, 'R'> const &h) { return make_pycapsule(handle<T, 'S'>{h}); }

  template <typename T> PyObject *make_pycapsule(handle<'B'> const &h) = delete; // Can not return a borrowed view to Python

} // namespace nda
