// Copyright (c) 2019-2021 Simons Foundation
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
#include <string_view>
#include "nda/exceptions.hpp"
#include "nda/traits.hpp"
#include "nda/declarations.hpp"
#include "nda/mem/address_space.hpp"
#include "nda/map.hpp"
#include "nda/mapped_functions.hpp"
#include "nda/mapped_functions.hxx"

#ifndef NDA_HAVE_DEVICE
#include "../device.hpp"
#endif

#if defined(NDA_HAVE_CUTENSOR)
#include "interface/cutensor_interface.hpp"
#endif

namespace nda::tensor {

  template <MemoryArray A>
  requires(is_blas_lapack_v<get_value_t<A>>) void scale(get_value_t<A> alpha, A &&a, op::TENSOR_OP oper = op::ID,
                                                        [[maybe_unused]] devStream_t const stream = 0) {

    if constexpr (mem::on_host<A>) {
      switch (oper) {
        case op::ID: a() *= alpha; break;
        case op::CONJ: a = nda::conj(a) * alpha; break;
        case op::SQRT: a = nda::sqrt(a) * alpha; break;
        case op::ABS: a = nda::abs(a) * alpha; break;
        case op::NEG: a = a * (-1.0 * alpha); break;
        default: NDA_RUNTIME_ERROR << "Unknown unary operation.";
      };
    } else { // on device
#if defined(NDA_HAVE_CUTENSOR)
      cutensor::cutensor_desc<get_value_t<A>, get_rank<A>> a_t(a);
      std::string indx = default_index<uint8_t(get_rank<A>)>();
      cutensor::permute(alpha, a_t, oper, a.data(), indx, a_t, a.data(), indx, stream);
#else
      compile_error_no_gpu();
#endif
    }
  }

} // namespace nda::tensor
