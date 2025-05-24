// Copyright (c) 2019-2023 Simons Foundation
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

#include <cstdlib>
#include <string>
#include "cutensor.h"

#include "nda/macros.hpp"
#include "nda/exceptions.hpp"
#include "cuda_runtime.h"

#include "nda/tensor/interface/cutensor_interface.hpp"

// use by default for now...
//#define USE_CUTENSOR_CACHE

namespace nda::tensor::cutensor {

  cutensorHandle_t &get_handle_ptr() {
    struct handle_t {
      handle_t() {
        cutensorCreate(&h);
#if defined(USE_CUTENSOR_CACHE)
        constexpr int32_t numCachelines = 1024;
        const size_t sizeCache          = numCachelines * sizeof(cutensorPlanCacheline_t);
        cachelines                      = (cutensorPlanCacheline_t *)malloc(sizeCache);
        CUTENSOR_CHECK(cutensorHandleAttachPlanCachelines, &h, cachelines, numCachelines);
#endif
      }
      ~handle_t() {
#if defined(USE_CUTENSOR_CACHE)
        CUTENSOR_CHECK(cutensorHandleDetachPlanCachelines, &h);
        free(cachelines);
#endif
        CUTENSOR_CHECK(cutensorDestroy, h);
      }

      cutensorHandle_t h = {};

      private:
#if defined(USE_CUTENSOR_CACHE)
      cutensorPlanCacheline_t *cachelines;
#endif
    };
    static handle_t h = {};
    return h.h;
  }

  // control device synchronization during cutensor calls 
  bool synchronize = true;
  bool get_synchronization() {return synchronize;}
  void set_synchronization(bool s_) {synchronize = s_;}

} // namespace nda::tensor::cutensor
