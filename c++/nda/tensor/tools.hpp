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

#if defined(NDA_HAVE_CUDA)
#include <cuda_runtime.h>
#endif

/// Tensor operations Interface
namespace nda::tensor 
{
  enum backend {__TBLIS__, __CUTENSOR__,__NONE__};

  namespace op {
    enum TENSOR_OP {ID,CONJ,SQRT,ABS,NEG,SUM,MUL,MAX,MIN};
  }

  // can I make this constexpr??? 
  template<uint8_t N>
  std::string default_index()
  {
    std::string indx{size_t(N)};
    for(uint8_t i=0; i<N; i++) indx[i] = static_cast<char>(i);
    return indx;
  }

  // contraction_plan_t = std::variant<dummy_contract_t,cutensor::contract_plan_t>;
}

