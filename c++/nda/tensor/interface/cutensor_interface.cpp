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

// use by default for now...
//#define USE_CUTENSOR_CACHE

namespace nda::tensor::cutensor {

  cutensorHandle_t* get_handle_ptr() {
    struct handle_t {
      handle_t() 
      { 
        cutensorInit(&h); 
#if defined(USE_CUTENSOR_CACHE)
        constexpr int32_t numCachelines = 1024;
        const size_t sizeCache = numCachelines * sizeof(cutensorPlanCacheline_t);
        cachelines = (cutensorPlanCacheline_t*) malloc(sizeCache);
        auto err = cutensorHandleAttachPlanCachelines(&h, cachelines, numCachelines);
        cudaDeviceSynchronize();
        if (err != CUTENSOR_STATUS_SUCCESS) NDA_RUNTIME_ERROR << std::string("cutensorHandleAttachPlanCachelines failed with error code ") + std::to_string(err) <<", "  <<cutensorGetErrorString(err); 
#endif
      }
      ~handle_t() 
      {  
#if defined(USE_CUTENSOR_CACHE)
	auto err = cutensorHandleDetachPlanCachelines(&h);
        cudaDeviceSynchronize();
        if (err != CUTENSOR_STATUS_SUCCESS) NDA_RUNTIME_ERROR << "cutensorHandleDetachPlanCachelines failed with error code "  <<std::to_string(err) <<", " <<cutensorGetErrorString(err); 
        free(cachelines);
#endif
      }
      cutensorHandle_t* get() { return std::addressof(h); }

      private:
      cutensorHandle_t h = {};
#if defined(USE_CUTENSOR_CACHE)
      cutensorPlanCacheline_t* cachelines;
#endif
    };
    static handle_t h = {};
    return h.get();
  }

  // always synchronize for now
  //static bool synchronize = true;

} // namespace nda::tensor::cutensor
