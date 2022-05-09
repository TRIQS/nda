// Copyright (c) 2019-2022 Simons Foundation
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

#pragma once
#include <complex>

#include "basic_array.hpp"
#include "blas/tools.hpp"
#include "lapack/interface/lapack_cxx_interface.hpp"

/// LAPACK Interface
namespace nda::lapack {
  using blas::has_C_layout;
  using blas::has_F_layout;
  using blas::get_ld;
} // namespace nda::lapack

#include "lapack/gelss.hpp"
#include "lapack/gesvd.hpp"
#include "lapack/getrf.hpp"
#include "lapack/getri.hpp"
#include "lapack/getrs.hpp"
#include "lapack/gtsv.hpp"
