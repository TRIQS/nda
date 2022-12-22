// Copyright (c) 2018-2021 Simons Foundation
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
// Authors: Miguel Morales

#pragma once

#include <iostream>
#include "../macros.hpp"
#if defined(NDA_HAVE_CUDA)
#include <cuda_runtime.h>

namespace nda::mem {

inline void device_check(cudaError_t sucess, std::string message = "")
{
  if (sucess != cudaSuccess) {
    if(message != std::string("")) std::cerr << message <<"\n";
    std::cerr << "Cuda runtime error: " <<std::to_string(sucess) <<" \n"
	      << "   Error name: " <<cudaGetErrorName(sucess) <<"\n"
              << "   Error message: " <<cudaGetErrorString(sucess) <<std::endl; 
    // MAM: terminate is not ideal in MPI environment, 
    //      maybe the library should have a user-defined terminate routine
    std::terminate();  
  }
}

} // nda::mem

#endif

