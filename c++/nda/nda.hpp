/*******************************************************************************
 *
 * TRIQS: a Toolbox for Research in Interacting Quantum Systems
 *
 * Copyright (C) 2011 by O. Parcollet
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

// FIXME : REMOVE THIS ?
// for python code generator, we need to know what to include...
#define TRIQS_INCLUDED_ARRAYS

#include "std_addons/complex.hpp"

// The basic classes
#include "basic_array_view.hpp"
#include "basic_array.hpp"

//#include "free_functions.hpp"
#include "basic_functions.hpp"
#include "functions.hpp"

#include "arithmetic.hpp"

#include "map.hpp"
#include "mapped_functions.hpp"

#include "algorithms.hpp"

#include <nda/print.hpp>

// HDF5 interface
//#include <nda/h5/simple_read_write.hpp>
//#include <nda/h5/array_of_non_basic.hpp>

// Regrouping indices
//#include <nda/group_indices.hpp>

// Reinterpretation of nx1x1 array and co
//#include <nda/reinterpret.hpp>

// Linear algebra ?? Keep here ?
//#include <nda/linalg/det_and_inverse.hpp>

//#include <nda/mpi.hpp>
