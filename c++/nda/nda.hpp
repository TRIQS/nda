// Copyright (c) 2019--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Includes all relevant headers for the core nda library.
 */

#pragma once

#ifdef NDA_DEBUG
#define NDA_ENFORCE_BOUNDCHECK
#endif

// FIXME : REMOVE THIS ?
// for python code generator, we need to know what to include...
#define TRIQS_INCLUDED_ARRAYS

#include "./accessors.hpp"
#include "./algorithms.hpp"
#include "./arithmetic.hpp"
#include "./array_adapter.hpp"
#include "./basic_array_view.hpp"
#include "./basic_array.hpp"
#include "./basic_functions.hpp"
#include "./clef.hpp"
#include "./concepts.hpp"
#include "./declarations.hpp"
#include "./device.hpp"
#include "./exceptions.hpp"
#include "./group_indices.hpp"
#include "./iterators.hpp"
#include "./layout_transforms.hpp"
#include "./layout.hpp"
#include "./linalg.hpp"
#include "./macros.hpp"
#include "./map.hpp"
#include "./mapped_functions.hpp"
#include "./mapped_functions.hxx"
#include "./matrix_functions.hpp"
#include "./mem.hpp"
#include "./print.hpp"
#include "./stdutil.hpp"
#include "./traits.hpp"

// If we are using c2py, include converters automatically
#ifdef C2PY_INCLUDED
#include <nda_py/c2py_converters.hpp>
#endif
