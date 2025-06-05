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

#include <mpi/mpi.hpp>
#include "nda/concepts.hpp"
#include "nda/traits.hpp"
#include "nda/macros.hpp"
#include "nda/exceptions.hpp"
#include "nda/device.hpp"
#include "nda/tensor/tools.hpp"
#include "cuda_runtime.h"

namespace nda::tensor::cutensor {

  // defined in cutensor_interface.cpp
  cutensorHandle_t &get_handle_ptr();

  // Global option to turn on/off the cudaDeviceSynchronize after cutensor library calls.
  extern bool synchronize; // NOLINT  (global option is on purpose)
  bool get_synchronization();
  void set_synchronization(bool s_);

#define CUTENSOR_CHECK(X, ...)                                                                                                                       \
  {                                                                                                                                                  \
    auto err = X(__VA_ARGS__);                                                                                                                       \
    if (err != CUTENSOR_STATUS_SUCCESS) {                                                                                                            \
      std::cerr << AS_STRING(X) << " failed with error code: " << std::to_string(err) << ", error message: " << cutensorGetErrorString(err)          \
                << std::endl;                                                                                                                        \
      mpi::communicator{}.abort(11);                                                                                                                 \
    }                                                                                                                                                \
    if (synchronize) {                                                                                                                               \
      auto errsync = cudaDeviceSynchronize();                                                                                                        \
      if (errsync != cudaSuccess) {                                                                                                                  \
        std::cerr << " cudaDeviceSynchronize failed after call to: " << AS_STRING(X) " \n "                                                          \
                  << " cudaGetErrorName: " << std::string(cudaGetErrorName(errsync)) << "\n"                                                         \
                  << " cudaGetErrorString: " << std::string(cudaGetErrorString(errsync)) << "\n";                                                    \
        mpi::communicator{}.abort(11);                                                                                                               \
      }                                                                                                                                              \
    }                                                                                                                                                \
  }

#define CUTENSOR_CHECK_STREAM(X, S, ...)                                                                                                             \
  {                                                                                                                                                  \
    auto err = X(__VA_ARGS__);                                                                                                                       \
    if (err != CUTENSOR_STATUS_SUCCESS) {                                                                                                            \
      std::cerr << AS_STRING(X) << " failed with error code: " << std::to_string(err) << ", error message: " << cutensorGetErrorString(err)          \
                << std::endl;                                                                                                                        \
      mpi::communicator{}.abort(11);                                                                                                                 \
    }                                                                                                                                                \
    if (synchronize) {                                                                                                                               \
      auto errsync = cudaStreamSynchronize(S);                                                                                                       \
      if (errsync != cudaSuccess) {                                                                                                                  \
        std::cerr << " cudaStreamSynchronize failed after call to: " << AS_STRING(X) " \n "                                                          \
                  << " cudaGetErrorName: " << std::string(cudaGetErrorName(errsync)) << "\n"                                                         \
                  << " cudaGetErrorString: " << std::string(cudaGetErrorString(errsync)) << "\n";                                                    \
        mpi::communicator{}.abort(11);                                                                                                               \
      }                                                                                                                                              \
    }                                                                                                                                                \
  }

  // cutensorOperator_t
  cutensorOperator_t constexpr cutensor_op(op::TENSOR_OP const oper) {
    cutensorOperator_t o = CUTENSOR_OP_UNKNOWN;
    switch (oper) {
      case op::SUM: return CUTENSOR_OP_ADD;
      case op::MUL: return CUTENSOR_OP_MUL;
      case op::MAX: return CUTENSOR_OP_MAX;
      case op::MIN: return CUTENSOR_OP_MIN;
      case op::ID: return CUTENSOR_OP_IDENTITY;
      case op::CONJ: return CUTENSOR_OP_CONJ;
      case op::SQRT: return CUTENSOR_OP_SQRT;
      case op::ABS: return CUTENSOR_OP_ABS;
      case op::NEG: return CUTENSOR_OP_NEG;
      default: static_assert(always_true<bool>, "Unknown cutensor operation."); return CUTENSOR_OP_UNKNOWN;
    };
    return o;
  };

  //cudaDataType_t
  template <typename T>
  auto data_type = std::enable_if_t<sizeof(T *) == 0>{};
  template <>
  inline auto data_type<float> = CUTENSOR_R_32F;
  template <>
  inline auto data_type<double> = CUTENSOR_R_64F;
  template <>
  inline auto data_type<std::complex<float>> = CUTENSOR_C_32F;
  template <>
  inline auto data_type<std::complex<double>> = CUTENSOR_C_64F;

  template <typename T>
  auto compute_type = std::enable_if_t<sizeof(T *) == 0>{};
  template <>
  inline auto compute_type<float> = CUTENSOR_COMPUTE_DESC_32F;
  template <>
  inline auto compute_type<double> = CUTENSOR_COMPUTE_DESC_64F;
  template <>
  inline auto compute_type<std::complex<float>> = CUTENSOR_COMPUTE_DESC_32F;
  template <>
  inline auto compute_type<std::complex<double>> = CUTENSOR_COMPUTE_DESC_64F;

  // define a compute_type that takes 3 types and a bool and returns the appropriate compute type, for the bool set to true it will lead to the "fast" version using e.g. CUTENSOR_COMPUTE_TF32 and CUTENSOR_COMPUTE_32F in case of double precision calculations

  template <typename ValueType, int Rank>
  struct cutensor_desc {
    static constexpr int rank = Rank;
    using value_type          = ValueType;

    // MAM: out of safety, since I don't know how cutensor operates
    std::array<long, rank> lens_, strides_;
    uint32_t alignment_              = 256; // MAM: need to get this dynamically somehow...
    cutensorTensorDescriptor_t desc_ = {};

    cutensor_desc() = delete;
    template <::nda::MemoryArrayOfRank<Rank> Arr>
      requires(Rank > 0)
    cutensor_desc(Arr const &a) : lens_(a.shape()), strides_(a.strides()) {
      CUTENSOR_CHECK(cutensorCreateTensorDescriptor, get_handle_ptr(), &desc_, uint32_t(Rank), lens_.data(), strides_.data(), data_type<ValueType>,
                     alignment_);
    }
    cutensor_desc(ValueType *) : lens_{}, strides_{} {
      static_assert(Rank == 0, "Rank mismatch.");
      CUTENSOR_CHECK(cutensorCreateTensorDescriptor, get_handle_ptr(), &desc_, uint32_t(Rank), NULL, NULL, data_type<ValueType>, alignment_);
    }
    ~cutensor_desc() { cutensorDestroyTensorDescriptor(desc_); }

    cutensor_desc(cutensor_desc const &) = delete;
    cutensor_desc(cutensor_desc &&other) = default;

    uint32_t alignment() const { return alignment_; }
    cutensorTensorDescriptor_t &desc() { return desc_; }
    cutensorTensorDescriptor_t const &desc() const { return desc_; }
  };

  struct cutensor_plan_t {
    /* default constructor */
    cutensor_plan_t() = default;

    /* construct using an operation */
    cutensor_plan_t(cutensorOperationDescriptor_t &desc, bool alloc = true, cutensorAlgo_t const algo = CUTENSOR_ALGO_DEFAULT,
                    cutensorWorksizePreference_t const workspacePref = CUTENSOR_WORKSPACE_DEFAULT)
       : plan{}, workspaceSizeEstimate(0), worksize(0), work(nullptr) {
      // Set the algorithm to use, no JIT yet!
      cutensorPlanPreference_t planPref;
      CUTENSOR_CHECK(cutensorCreatePlanPreference, get_handle_ptr(), &planPref, algo, CUTENSOR_JIT_MODE_NONE);

      CUTENSOR_CHECK(cutensorEstimateWorkspaceSize, get_handle_ptr(), desc, planPref, workspacePref, &workspaceSizeEstimate);

      // Create Contraction Plan
      CUTENSOR_CHECK(cutensorCreatePlan, get_handle_ptr(), std::addressof(plan), desc, planPref, workspaceSizeEstimate);

      if (alloc) {
        uint64_t actualWorkspaceSize = 0;
        CUTENSOR_CHECK(cutensorPlanGetAttribute, get_handle_ptr(), plan, CUTENSOR_PLAN_REQUIRED_WORKSPACE, &actualWorkspaceSize,
                       sizeof(actualWorkspaceSize));
        resize_workspace_if_needed(workspaceSizeEstimate);
      }
    }

    ~cutensor_plan_t() { clear(); }

    cutensor_plan_t(cutensor_plan_t const &other) = delete;
    cutensor_plan_t(cutensor_plan_t &&other)
       : plan(other.plan), workspaceSizeEstimate(other.workspaceSizeEstimate), worksize(other.worksize), work(other.work) {
      other.work = nullptr;
      other.clear();
    }

    void clear() {
      if (work != nullptr) cudaFree(work);
      cutensorDestroyPlan(plan);
      work                  = nullptr;
      worksize              = 0;
      workspaceSizeEstimate = 0;
    }

    uint64_t get_workspace_size() const { return worksize; }
    void *get_workspace() { return work; }

    void resize_workspace_if_needed(uint64_t size) {
      if (size > worksize) {
        worksize = size;
        if (work != nullptr) cudaFree(work);
        work = nullptr;
        device_error_check(cudaMalloc((void **)&work, worksize), "cudaMalloc");
      }
    }

    cutensorPlan_t plan            = {};
    uint64_t workspaceSizeEstimate = 0;

    private:
    uint64_t worksize = 0;
    void *work        = nullptr;
  };

  /*************************************************************************
   *                            contraction                                *
   ************************************************************************/

  template <typename value_t, int rA, int rB, int rC>
    requires(rA > 0 and rB > 0 and rC >= 0)
  void contract(value_t alpha, cutensor_desc<value_t, rA> const &descA, op::TENSOR_OP op_A, value_t const *A_d, std::string_view idxA,
                cutensor_desc<value_t, rB> const &descB, op::TENSOR_OP op_B, value_t const *B_d, std::string_view idxB, value_t beta,
                cutensor_desc<value_t, rC> &descC, op::TENSOR_OP op_C, value_t *C_d, std::string_view idxC, cudaStream_t const stream = 0) {
    std::array<int, rA> modeA;
    std::array<int, rB> modeB;
    std::array<int, rC> modeC;
    std::copy_n(idxA.begin(), rA, modeA.begin());
    std::copy_n(idxB.begin(), rB, modeB.begin());
    if constexpr (rC > 0) std::copy_n(idxC.begin(), rC, modeC.begin());

    // Create the Contraction Descriptor
    cutensorOperationDescriptor_t desc;
    int *modeC_data = (rC > 0 ? modeC.data() : NULL);
    CUTENSOR_CHECK(cutensorCreateContraction, get_handle_ptr(), &desc, descA.desc(), modeA.data(), cutensor_op(op_A), descB.desc(), modeB.data(),
                   cutensor_op(op_B), descC.desc(), modeC_data, cutensor_op(op_C), descC.desc(), modeC_data, compute_type<value_t>);

    cutensor_plan_t plan(desc, true);
    CUTENSOR_CHECK_STREAM(cutensorContract, stream, get_handle_ptr(), plan.plan, (void *)&alpha, A_d, B_d, (void *)&beta, C_d, C_d,
                          plan.get_workspace(), plan.get_workspace_size(), stream);
  }

  /*************************************************************************
   *                            elementwise binary                         *
   ************************************************************************/

  template <typename value_t, int rA, int rB>
    requires(rA >= 0 and rB > 0 and rB >= rA)
  void elementwise_binary(value_t const alpha, cutensor_desc<value_t, rA> const &descA, op::TENSOR_OP op_A, value_t const *A_d, std::string_view idxA,
                          value_t const gamma, cutensor_desc<value_t, rB> const &descB, op::TENSOR_OP op_B, value_t const *B_d, std::string_view idxB,
                          value_t *C_d, op::TENSOR_OP oper, cudaStream_t const stream = 0) {
    std::array<int, rB> modeB;
    std::copy_n(idxB.begin(), rB, modeB.begin());

    if constexpr (rA > 0) {

      std::array<int, rA> modeA;
      std::copy_n(idxA.begin(), rA, modeA.begin());

      cutensorOperationDescriptor_t desc;
      CUTENSOR_CHECK(cutensorCreateElementwiseBinary, get_handle_ptr(), &desc, descA.desc(), modeA.data(), cutensor_op(op_A), descB.desc(),
                     modeB.data(), cutensor_op(op_B), descB.desc(), modeB.data(), cutensor_op(oper), compute_type<value_t>);

      cutensor_plan_t plan(desc, false);
      CUTENSOR_CHECK_STREAM(cutensorElementwiseBinaryExecute, stream, get_handle_ptr(), plan.plan, (const void *)&alpha, A_d, (void *)&gamma, B_d,
                            C_d, stream);

    } else {

      cutensorOperationDescriptor_t desc;
      CUTENSOR_CHECK(cutensorCreateElementwiseBinary, get_handle_ptr(), &desc, descA.desc(), NULL, cutensor_op(op_A), descB.desc(), modeB.data(),
                     cutensor_op(op_B), descB.desc(), modeB.data(), cutensor_op(oper), compute_type<value_t>);

      cutensor_plan_t plan(desc, false);
      CUTENSOR_CHECK_STREAM(cutensorElementwiseBinaryExecute, stream, get_handle_ptr(), plan.plan, (const void *)&alpha, A_d, (void *)&gamma, B_d,
                            C_d, stream);
    }
  }

  /*************************************************************************
   *                              permute                                  *
   ************************************************************************/

  // MAM: this routine could be used to convert value_types, generalized later! need
  //      new data_type<typeA,typeB> with allowed combinations...
  template <typename value_t, int rank>
  void permute(value_t const alpha, cutensor_desc<value_t, rank> const &descA, op::TENSOR_OP op_A, value_t const *A_d, std::string_view const idxA,
               cutensor_desc<value_t, rank> const &descB, value_t *B_d, std::string_view const idxB, cudaStream_t const stream = 0) {
    std::array<int, rank> modeA;
    std::array<int, rank> modeB;
    std::copy_n(idxA.begin(), rank, modeA.begin());
    std::copy_n(idxB.begin(), rank, modeB.begin());

    cutensorOperationDescriptor_t desc;
    CUTENSOR_CHECK(cutensorCreatePermutation, get_handle_ptr(), &desc, descA.desc(), modeA.data(), cutensor_op(op_A), descB.desc(), modeB.data(),
                   compute_type<value_t>);

    cutensor_plan_t plan(desc, false);
    CUTENSOR_CHECK_STREAM(cutensorPermute, stream, get_handle_ptr(), plan.plan, (const void *)&alpha, A_d, B_d, stream);
  }

  /*************************************************************************
   *                              reduce                                   *
   ************************************************************************/

  template <typename value_t, int rA, int rB>
  void reduce(value_t const alpha, cutensor_desc<value_t, rA> const &descA, op::TENSOR_OP op_A, value_t const *A_d, std::string_view const idxA,
              value_t beta, cutensor_desc<value_t, rB> const &descB, op::TENSOR_OP op_B, value_t const *B_d, std::string_view const idxB,
              value_t *C_d, op::TENSOR_OP oper, cudaStream_t const stream = 0) {
    std::array<int, rA> modeA;
    std::array<int, rB> modeB;
    std::copy_n(idxA.begin(), rA, modeA.begin());
    std::copy_n(idxB.begin(), rB, modeB.begin());

    cutensorOperationDescriptor_t desc;
    CUTENSOR_CHECK(cutensorCreateReduction, get_handle_ptr(), &desc, descA.desc(), modeA.data(), cutensor_op(op_A), descB.desc(), modeB.data(),
                   cutensor_op(op_B), descB.desc(), modeB.data(), cutensor_op(oper), compute_type<value_t>);

    cutensor_plan_t plan(desc, true);
    CUTENSOR_CHECK_STREAM(cutensorReduce, stream, get_handle_ptr(), plan.plan, (const void *)&alpha, A_d, (const void *)&beta, B_d, C_d,
                          plan.get_workspace(), plan.get_workspace_size(), stream);
  }

  template <typename value_t, int rA>
  void reduce(value_t const alpha, cutensor_desc<value_t, rA> const &descA, op::TENSOR_OP op_A, value_t const *A_d, std::string_view const idxA,
              value_t *C_d, op::TENSOR_OP oper, cudaStream_t const stream = 0) {
    value_t beta(0);
    std::array<int, rA> modeA;
    std::copy_n(idxA.begin(), rA, modeA.begin());

    uint32_t alignment_ = 256;
    cutensorTensorDescriptor_t Tdesc_;
    CUTENSOR_CHECK(cutensorCreateTensorDescriptor, get_handle_ptr(), &Tdesc_, 0, NULL, NULL, data_type<value_t>, alignment_);

    cutensorOperationDescriptor_t Opdesc;
    CUTENSOR_CHECK(cutensorCreateReduction, get_handle_ptr(), &Opdesc, descA.desc(), modeA.data(), cutensor_op(op_A), Tdesc_, nullptr,
                   cutensor_op(op::ID), Tdesc_, nullptr, cutensor_op(oper), compute_type<value_t>);

    cutensor_plan_t plan(Opdesc, true);
    CUTENSOR_CHECK_STREAM(cutensorReduce, stream, get_handle_ptr(), plan.plan, (const void *)&alpha, A_d, (const void *)&beta, C_d, C_d,
                          plan.get_workspace(), plan.get_workspace_size(), stream);
    cutensorDestroyTensorDescriptor(Tdesc_);
  }

} // namespace nda::tensor::cutensor
