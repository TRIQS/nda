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

#pragma once
#include <cstdlib>
#include <string>
#include <vector>
#include <complex>
#include "cutensor.h"

#include "nda/concepts.hpp"
#include "nda/macros.hpp"
#include "nda/exceptions.hpp"
#include "nda/mem/device.hpp"
#include "nda/tensor/tools.hpp"
#include "cuda_runtime.h"

namespace nda::tensor::cutensor {

  // defined in cutensor_interface.cpp 
  cutensorHandle_t* get_handle_ptr();
  
// MAM: always synchronize for now...
#define CUTENSOR_CHECK(X, ...)               \
  { auto err = X(__VA_ARGS__);   \
  cudaDeviceSynchronize();       \
  if (err != CUTENSOR_STATUS_SUCCESS) NDA_RUNTIME_ERROR << AS_STRING(X) <<" failed with error code " <<std::to_string(err) <<", " <<cutensorGetErrorString(err); }

  // cutensorOperator_t 
  cutensorOperator_t constexpr cutensor_op(op::TENSOR_OP const oper) {
    cutensorOperator_t o = CUTENSOR_OP_UNKNOWN;
    switch(oper) {
      case op::SUM:
        return CUTENSOR_OP_ADD;
      case op::MAX:
        return CUTENSOR_OP_MAX;
      case op::MIN:
        return CUTENSOR_OP_MIN;
      case op::ID:
        return CUTENSOR_OP_IDENTITY;
      case op::CONJ:
        return CUTENSOR_OP_CONJ;
      case op::SQRT:
        return CUTENSOR_OP_SQRT;
      case op::ABS:
        return CUTENSOR_OP_ABS;
      case op::NEG:
        return CUTENSOR_OP_NEG;
      default:
        return CUTENSOR_OP_UNKNOWN;
    };
    return o;
  };

  //cudaDataType_t 
  template<typename T> auto data_type = std::enable_if_t<sizeof(T*)==0>{};
  template<> inline auto data_type<float>                = CUDA_R_32F;    
  template<> inline auto data_type<double>               = CUDA_R_64F;    
  template<> inline auto data_type<std::complex<float>>  = CUDA_C_32F;    
  template<> inline auto data_type<std::complex<double>> = CUDA_C_64F;    

  template<typename T> auto compute_type = std::enable_if_t<sizeof(T*)==0>{};
  template<> inline auto compute_type<float>                = CUTENSOR_COMPUTE_32F;
  template<> inline auto compute_type<double>               = CUTENSOR_COMPUTE_64F;
  template<> inline auto compute_type<std::complex<float>>  = CUTENSOR_COMPUTE_32F;
  template<> inline auto compute_type<std::complex<double>> = CUTENSOR_COMPUTE_64F;

  // define a compute_type that takes 3 types and a bool and returns the appropriate compute type, for the bool set to true it will lead to the "fast" version using e.g. CUTENSOR_COMPUTE_TF32 and CUTENSOR_COMPUTE_32F in case of double precision calculations

  template<typename ValueType, int Rank>
  struct cutensor_desc {
    static constexpr int rank = Rank;
    using value_type = ValueType;

    // MAM: out of safety, since I don't know how cutensor operates
    std::array<long, rank> lens_, strides_;
    uint32_t alignment_ = 0;
    cutensorTensorDescriptor_t desc_ = {};

    cutensor_desc() = delete;
    template<::nda::MemoryArrayOfRank<Rank> Arr>
    requires( Rank > 0 )
    cutensor_desc(Arr const& a, op::TENSOR_OP const oper = op::ID) :
      lens_(a.shape()),
      strides_(a.strides())
    {
      CUTENSOR_CHECK(cutensorInitTensorDescriptor, get_handle_ptr(), &desc_, uint32_t(Rank), 
  		     lens_.data(), strides_.data(), data_type<ValueType>, cutensor_op(oper)); 
      CUTENSOR_CHECK(cutensorGetAlignmentRequirement, get_handle_ptr(), (const void *)a.data(),
		     &desc_, &alignment_); 
    }
    cutensor_desc(ValueType *a, op::TENSOR_OP const oper = op::ID) :
      lens_{},strides_{}
    {
      static_assert( Rank==0, "Rank mismatch.");
      CUTENSOR_CHECK(cutensorInitTensorDescriptor, get_handle_ptr(), &desc_, uint32_t(Rank),
                     NULL, NULL, data_type<ValueType>, cutensor_op(oper));
      CUTENSOR_CHECK(cutensorGetAlignmentRequirement, get_handle_ptr(), (void *)a,
                     &desc_, &alignment_);
    }
    ~cutensor_desc() = default;

    cutensor_desc(cutensor_desc const&) = delete;
    cutensor_desc(cutensor_desc&& other) = default;

    uint32_t alignment() const { return alignment_; }
    cutensorTensorDescriptor_t* desc() { return std::addressof(desc_); }
    cutensorTensorDescriptor_t const* desc() const { return std::addressof(desc_); }
  };

  /*************************************************************************
   *                            contraction                                *
   ************************************************************************/

  struct contract_plan_t
  {
    contract_plan_t() { plan = new cutensorContractionPlan_t{}; }
    ~contract_plan_t() { clear(); }

    // what should I do here?
    contract_plan_t(contract_plan_t const& other) = delete;

    contract_plan_t(contract_plan_t && other) :
      plan(other.plan),worksize(other.worksize),work(other.work)
    {
      other.plan = nullptr;
      other.worksize = 0;
      other.work = nullptr;
    }

    void clear() {
      if(plan != nullptr) delete plan;
      if(work != nullptr) cudaFree(work);
      plan = nullptr;
      worksize=0;
      work=nullptr;
    }

    cutensorContractionPlan_t* plan = nullptr;
    size_t worksize = 0;
    void* work = nullptr;
  };

  template<typename value_t, int rA, int rB, int rC>
  requires( rA>0 and rB>0 and rC>=0 )
  contract_plan_t create_contract_plan(
	    cutensor_desc<value_t,rA> const& descA, std::string_view idxA,
            cutensor_desc<value_t,rB> const& descB, std::string_view idxB,
            cutensor_desc<value_t,rC> & descC, std::string_view idxC,
	    bool alloc = true)
  { 
    std::array<int,rA> modeA;	 
    std::array<int,rB> modeB;	 
    std::array<int,rC> modeC;
    std::copy_n(idxA.begin(),rA,modeA.begin());	 
    std::copy_n(idxB.begin(),rB,modeB.begin());	 
    if constexpr (rC > 0) std::copy_n(idxC.begin(),rC,modeC.begin());	 

    // Create the Contraction Descriptor
    cutensorContractionDescriptor_t desc;
    int* modeC_data = (rC>0?modeC.data():NULL); 
    CUTENSOR_CHECK( cutensorInitContractionDescriptor, get_handle_ptr(), &desc,
              descA.desc(), modeA.data(), descA.alignment(),
              descB.desc(), modeB.data(), descB.alignment(),
              descC.desc(), modeC_data, descC.alignment(),
              descC.desc(), modeC_data, descC.alignment(),
              compute_type<value_t> );

    // Set the algorithm to use
    cutensorContractionFind_t find;
    CUTENSOR_CHECK( cutensorInitContractionFind, get_handle_ptr(), &find,
                  CUTENSOR_ALGO_DEFAULT) ;

    contract_plan_t plan;

    // Query workspace
    CUTENSOR_CHECK( cutensorContractionGetWorkspaceSize, get_handle_ptr(), &desc, &find,
                  CUTENSOR_WORKSPACE_RECOMMENDED, &plan.worksize ) ;

    if( alloc and plan.worksize > 0 )
      mem::device_check( cudaMalloc((void**) &plan.work, plan.worksize), "cudaMalloc" );

    // Create Contraction Plan
    CUTENSOR_CHECK( cutensorInitContractionPlan, get_handle_ptr(), plan.plan, &desc,
                                            &find, plan.worksize);

    return plan;
  }

  template<typename value_t>
  void contract(contract_plan_t &plan, 
		value_t alpha, value_t const* A_d, value_t const* B_d,
		value_t beta , value_t * C_d)
  {
    if(plan.work != nullptr)  {

      CUTENSOR_CHECK( cutensorContraction, get_handle_ptr(), plan.plan,
                      (void*)&alpha, A_d, B_d, (void*)&beta,  C_d, C_d,
                      plan.work, plan.worksize, 0 );

    } else {  

      void* work = nullptr;
      if(plan.worksize > 0)
        mem::device_check( cudaMalloc((void**) &work, plan.worksize), "cudaMalloc" );

      // Execute the tensor contraction
      CUTENSOR_CHECK( cutensorContraction, get_handle_ptr(), plan.plan,
                     (void*)&alpha, A_d, B_d, (void*)&beta,  C_d, C_d,
                     work, plan.worksize, 0 );

      if(plan.worksize > 0)
        cudaFree(work);
    }
  }

  template<typename value_t, int rA, int rB, int rC>
  requires( rA>0 and rB>0 and rC>=0 )
  void contract(value_t alpha,
            cutensor_desc<value_t,rA> const& descA, value_t const* A_d, std::string_view idxA,
            cutensor_desc<value_t,rB> const& descB, value_t const* B_d, std::string_view idxB,
            value_t beta,
            cutensor_desc<value_t,rC> & descC, value_t * C_d, std::string_view idxC)
  {
    // don't allocate during plan creation when buffer is ready, allocate on the fly on contract!
    auto plan = create_contract_plan(descA,idxA,descB,idxB,descC,idxC,true);
    contract(plan,alpha,A_d,B_d,beta,C_d);
  }


  /*************************************************************************
   *                            elementwise binary                         *
   ************************************************************************/

  template<typename value_t, int rA, int rB>
  requires( rA >= 0 and rB > 0 and rB >= rA )
  void elementwise_binary(value_t alpha, 
	cutensor_desc<value_t,rA> const& descA, value_t const* A_d, std::string_view idxA,
        value_t gamma,
        cutensor_desc<value_t,rB> const& descB, value_t const* B_d, std::string_view idxB,
        value_t * C_d, op::TENSOR_OP oper)
  {
    std::array<int,rB> modeB;
    std::copy_n(idxB.begin(),rB,modeB.begin());

    if constexpr (rA > 0) {
      std::array<int,rA> modeA;
      std::copy_n(idxA.begin(),rA,modeA.begin());
      CUTENSOR_CHECK( cutensorElementwiseBinary, get_handle_ptr(), (void*) &alpha, 
	A_d, descA.desc(), modeA.data(), (void*) &gamma, B_d, descB.desc(), modeB.data(),	 
	C_d, descB.desc(), modeB.data(), cutensor_op(oper), data_type<value_t>, 0);
    } else {
      CUTENSOR_CHECK( cutensorElementwiseBinary, get_handle_ptr(), (void*) &alpha, 
	A_d, descA.desc(), NULL, (void*) &gamma, B_d, descB.desc(), modeB.data(), 
	C_d, descB.desc(), modeB.data(), cutensor_op(oper), data_type<value_t>, 0);
    }

  }

  /*************************************************************************
   *                              permute                                  *
   ************************************************************************/

  // MAM: this routine could be used to convert value_types, generalized later! need
  //      new data_type<typeA,typeB> with allowed combinations...
  template<typename value_t, int rank>
  void permute(value_t alpha,
        cutensor_desc<value_t,rank> const& descA, value_t const* A_d, std::string_view const idxA,
        cutensor_desc<value_t,rank> const& descB, value_t * B_d, std::string_view const idxB)
  {
    std::array<int,rank> modeA;
    std::array<int,rank> modeB;
    std::copy_n(idxA.begin(),rank,modeA.begin());
    std::copy_n(idxB.begin(),rank,modeB.begin());

    CUTENSOR_CHECK( cutensorPermutation, get_handle_ptr(), (void*) &alpha,
        A_d, descA.desc(), modeA.data(), B_d, descB.desc(), modeB.data(),
        data_type<value_t>, 0);
  }

  /*************************************************************************
   *                              permute                                  *
   ************************************************************************/

  template<typename value_t, int rA, int rB>
  void reduce(value_t alpha,
        cutensor_desc<value_t,rA> const& descA, value_t const* A_d, std::string_view const idxA,
        value_t beta,
        cutensor_desc<value_t,rB> const& descB, value_t const* B_d, std::string_view const idxB,
        value_t * C_d, op::TENSOR_OP oper)
  {
    std::array<int,rA> modeA;
    std::array<int,rB> modeB;
    std::copy_n(idxA.begin(),rA,modeA.begin());
    std::copy_n(idxB.begin(),rB,modeB.begin());

    uint64_t workspaceSize = 0;
    CUTENSOR_CHECK( cutensorReductionGetWorkspaceSize, get_handle_ptr(), 
        A_d, descA.desc(), modeA.data(), B_d, descB.desc(), modeB.data(),
        C_d, descB.desc(), modeB.data(), cutensor_op(oper), 
	compute_type<value_t>, &workspaceSize);

    // MAM: use buffer!
    void* work;
    if( workspaceSize > 0 )
      mem::device_check( cudaMalloc((void**) &work, workspaceSize), "cudaMalloc" );

    CUTENSOR_CHECK( cutensorReduction, get_handle_ptr(), (void*) &alpha, 
        A_d, descA.desc(), modeA.data(), (void*) &beta, B_d, descB.desc(), modeB.data(),
        C_d, descB.desc(), modeB.data(), cutensor_op(oper), compute_type<value_t>, 
	work, workspaceSize, 0);

    if( workspaceSize > 0 )
      cudaFree(work);
  }

  template<typename value_t, int rA>
  void reduce(value_t alpha,
        cutensor_desc<value_t,rA> const& descA, value_t const* A_d, std::string_view const idxA,
        value_t * C_d, op::TENSOR_OP oper)
  {
    std::array<int,rA> modeA;
    std::copy_n(idxA.begin(),rA,modeA.begin());

    uint32_t alignment_ = 0;
    cutensorTensorDescriptor_t desc_;
    CUTENSOR_CHECK(cutensorInitTensorDescriptor, get_handle_ptr(), &desc_, 0,
                   NULL, NULL, data_type<value_t>, CUTENSOR_OP_IDENTITY);
    CUTENSOR_CHECK(cutensorGetAlignmentRequirement, get_handle_ptr(), (void *) C_d,
                   &desc_, &alignment_);

    uint64_t workspaceSize = 0;
    CUTENSOR_CHECK( cutensorReductionGetWorkspaceSize, get_handle_ptr(),
        A_d, descA.desc(), modeA.data(), C_d, &desc_, NULL,
        C_d, &desc_, NULL, cutensor_op(oper),
        compute_type<value_t>, &workspaceSize);

    // MAM: use buffer!
    void* work;
    if( workspaceSize > 0 )
      mem::device_check( cudaMalloc((void**) &work, workspaceSize), "cudaMalloc" );

    value_t beta{0}; 
    CUTENSOR_CHECK( cutensorReduction, get_handle_ptr(), (void*) &alpha,
        A_d, descA.desc(), modeA.data(), (void*) &beta, C_d, &desc_, NULL,
        C_d, &desc_, NULL, cutensor_op(oper), compute_type<value_t>,
        work, workspaceSize, 0);

    if( workspaceSize > 0 )
      cudaFree(work);
  }



  

} // namespace nda::tensor::cutensor
