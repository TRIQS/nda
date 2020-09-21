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

#pragma once

#ifdef NDA_DEBUG
#define NDA_ENFORCE_BOUNDCHECK
#endif

// FIXME : REMOVE THIS ?
// for python code generator, we need to know what to include...
#define TRIQS_INCLUDED_ARRAYS

#include "basic_array_view.hpp"
#include "basic_array.hpp"

#include "basic_functions.hpp"
#include "layout_transforms.hpp"

#include "array_adapter.hpp"

#include "matrix_functions.hpp"

#include "arithmetic.hpp"

#include "map.hpp"
#include "mapped_functions.hpp"
#include "mapped_functions.hxx"

#include "algorithms.hpp"
#include "print.hpp"

#include "layout/rect_str.hpp"

#include "backward.hpp"
