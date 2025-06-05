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

/// Tensor operations Interface
namespace nda::tensor {}

#include "blas/tools.hpp"
#include "tensor/tools.hpp"
#include "tensor/contract.hpp"
#include "tensor/add.hpp"
#include "tensor/set.hpp"
#include "tensor/scale.hpp"
#include "tensor/dot.hpp"
#include "tensor/reduce.hpp"
#include "tensor/assign.hpp"
#include "tensor/elementwise.hpp"

namespace nda::tensor {
#if defined(NDA_HAVE_CUTENSOR)
  inline bool get_device_synchronization() { return nda::tensor::cutensor::get_synchronization(); }
  inline void set_device_synchronization(bool s_) { nda::tensor::cutensor::set_synchronization(s_); }
#else
  inline static bool __synchronize__ = true;
  inline bool get_device_synchronization() { return __synchronize__; }
  inline void set_device_synchronization(bool s_) { __synchronize__ = s_; }
#endif
} // namespace nda::tensor
