/*******************************************************************************
 *
 * TRIQS: a Toolbox for Research in Interacting Quantum Systems
 *
 * Copyright (C) 2011-2013 by O. Parcollet
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

#include "./h5/file.hpp"
#include "./h5/group.hpp"
#include "./h5/scheme.hpp"
#include "./h5/scalar.hpp"
#include "./h5/generic.hpp"
#include "./h5/stl/string.hpp"
#include "./h5/stl/vector.hpp"
#include "./h5/stl/map.hpp"
#include "./h5/stl/pair.hpp"
#include "./h5/stl/tuple.hpp"
#include "./h5/stl/optional.hpp"

//#include "./h5/stl/variant.hpp"

// FIXME : Why all these include by default ?

// FIXME : Still needed ?
// for python code generator, we need to know what has to been included.
//#define TRIQS_INCLUDED_H5

// in some old version of hdf5 (Ubuntu 12.04 e.g.), the macro is not yet defined.
#ifndef H5_VERSION_GE

#define H5_VERSION_GE(Maj, Min, Rel)                                                                                                                 \
  (((H5_VERS_MAJOR == Maj) && (H5_VERS_MINOR == Min) && (H5_VERS_RELEASE >= Rel)) || ((H5_VERS_MAJOR == Maj) && (H5_VERS_MINOR > Min))               \
   || (H5_VERS_MAJOR > Maj))

#endif

