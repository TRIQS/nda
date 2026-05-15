// Copyright (c) 2024--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Implementation details for tensor/interface/cutensor_interface.hpp.
 */

#include "./cutensor_interface.hpp"
#include "../tools.hpp"
#include "../../concepts.hpp"
#include "../../device.hpp"
#include "../../exceptions.hpp"

#include <cutensor.h>

#include <algorithm>
#include <bit>
#include <complex>
#include <cstdint>
#include <string_view>
#include <type_traits>
#include <vector>

namespace nda::tensor::device {

  // File-local synchronization flag, exposed via the get/set functions below (matches the BLAS interface pattern).
  thread_local bool synchronize = true; // NOLINT (per-thread option is on purpose)
  void set_synchronization(bool do_sync) noexcept { synchronize = do_sync; }
  bool get_synchronization() noexcept { return synchronize; }

  // Get the cutensor handle.
  cutensorHandle_t &get_handle() {
    struct handle_storage_t { // RAII for handle
      handle_storage_t() { cutensorCreate(&handle); }
      ~handle_storage_t() { cutensorDestroy(handle); }
      cutensorHandle_t handle = {};
    };
    static auto sto = handle_storage_t{};
    return sto.handle;
  }

  // Anonymous namespace for local functions.
  namespace {

    // Check the success of a cutensor operation.
    void cutensor_error_check(cutensorStatus_t status, std::string_view func) {
      if (status != CUTENSOR_STATUS_SUCCESS) {
        NDA_RUNTIME_ERROR << "cuTENSOR runtime error in function " << func << "\n"
                          << " cutensorStatus_t: " << status << "\n"
                          << " cutensorGetErrorString: " << cutensorGetErrorString(status) << "\n";
      }
    }

    // Cuda data type conversion.
    template <typename T, typename U = std::remove_const_t<T>>
    constexpr auto cuda_data_type() {
      if constexpr (std::is_same_v<U, float>) {
        return CUTENSOR_R_32F;
      } else if constexpr (std::is_same_v<U, double>) {
        return CUTENSOR_R_64F;
      } else if constexpr (std::is_same_v<U, std::complex<float>>) {
        return CUTENSOR_C_32F;
      } else if constexpr (std::is_same_v<U, std::complex<double>>) {
        return CUTENSOR_C_64F;
      }
    }

    // Cutensor compute type conversion.
    template <typename T, typename U = std::remove_const_t<T>>
    constexpr auto cutensor_compute_type() {
      if constexpr (AnyOf<U, float, std::complex<float>>) {
        return CUTENSOR_COMPUTE_DESC_32F;
      } else if constexpr (AnyOf<U, double, std::complex<double>>) {
        return CUTENSOR_COMPUTE_DESC_64F;
      }
    }

    // Find the pointer alignment for a given pointer, capped at 256.
    template <typename T>
    auto find_alignment(T *p) {
      auto const x = reinterpret_cast<std::uintptr_t>(p); // NOLINT (reinterpret_cast is necessary here)
      // largest power-of-two divisor of the address, capped at 256 (cudaMalloc default alignment)
      std::uintptr_t alignment = std::uintptr_t(1) << std::countr_zero(x);
      return static_cast<std::uint32_t>(std::min(alignment, std::uintptr_t(256)));
    }

    // Convert an index string to a vector of int32_t mode labels for cuTENSOR.
    auto to_modes(std::string_view idx) { return std::vector<std::int32_t>(idx.begin(), idx.end()); }

    // Map unary_op enum to cuTENSOR unary operator.
    // clang-format off
    cutensorOperator_t to_cutensor_unary_op(unary_op op) {
      switch (op) {
        case unary_op::IDENTITY:  return CUTENSOR_OP_IDENTITY;
        case unary_op::SQRT:      return CUTENSOR_OP_SQRT;
        case unary_op::RELU:      return CUTENSOR_OP_RELU;
        case unary_op::CONJ:      return CUTENSOR_OP_CONJ;
        case unary_op::RCP:       return CUTENSOR_OP_RCP;
        case unary_op::SIGMOID:   return CUTENSOR_OP_SIGMOID;
        case unary_op::TANH:      return CUTENSOR_OP_TANH;
        case unary_op::EXP:       return CUTENSOR_OP_EXP;
        case unary_op::LOG:       return CUTENSOR_OP_LOG;
        case unary_op::ABS:       return CUTENSOR_OP_ABS;
        case unary_op::NEG:       return CUTENSOR_OP_NEG;
        case unary_op::SIN:       return CUTENSOR_OP_SIN;
        case unary_op::COS:       return CUTENSOR_OP_COS;
        case unary_op::TAN:       return CUTENSOR_OP_TAN;
        case unary_op::SINH:      return CUTENSOR_OP_SINH;
        case unary_op::COSH:      return CUTENSOR_OP_COSH;
        case unary_op::ASIN:      return CUTENSOR_OP_ASIN;
        case unary_op::ACOS:      return CUTENSOR_OP_ACOS;
        case unary_op::ATAN:      return CUTENSOR_OP_ATAN;
        case unary_op::ASINH:     return CUTENSOR_OP_ASINH;
        case unary_op::ACOSH:     return CUTENSOR_OP_ACOSH;
        case unary_op::ATANH:     return CUTENSOR_OP_ATANH;
        case unary_op::CEIL:      return CUTENSOR_OP_CEIL;
        case unary_op::FLOOR:     return CUTENSOR_OP_FLOOR;
        case unary_op::MISH:      return CUTENSOR_OP_MISH;
        case unary_op::SWISH:     return CUTENSOR_OP_SWISH;
        case unary_op::SOFT_PLUS: return CUTENSOR_OP_SOFT_PLUS;
        case unary_op::SOFT_SIGN: return CUTENSOR_OP_SOFT_SIGN;
        default: NDA_RUNTIME_ERROR << "nda::tensor::cutensor: unary_op has no cuTENSOR equivalent";
      }
    }
    // clang-format on

    // Map binary_op enum to cuTENSOR binary operator.
    cutensorOperator_t to_cutensor_binary_op(binary_op op) {
      switch (op) {
        case binary_op::SUM: return CUTENSOR_OP_ADD;
        case binary_op::PROD: return CUTENSOR_OP_MUL;
        case binary_op::MAX: return CUTENSOR_OP_MAX;
        case binary_op::MIN: return CUTENSOR_OP_MIN;
        default: NDA_RUNTIME_ERROR << "nda::tensor::cutensor: binary_op has no cuTENSOR equivalent";
      }
    }

    // Create a tensor descriptor from a tensor view.
    template <typename T>
    auto create_tensor_desc(tensor_view<T> tv) {
      cutensorTensorDescriptor_t desc{};
      auto status = cutensorCreateTensorDescriptor(get_handle(), &desc, static_cast<std::uint32_t>(tv.ndim), tv.extents, tv.strides,
                                                   cuda_data_type<T>(), find_alignment(tv.data));
      cutensor_error_check(status, "cutensorCreateTensorDescriptor");
      return desc;
    }

    // Destroy a given tensor descriptor.
    void destroy_tensor_desc(cutensorTensorDescriptor_t desc) {
      cutensor_error_check(cutensorDestroyTensorDescriptor(desc), "cutensorDestroyTensorDescriptor");
    }

    // Create a plan preference with default algorithm and no JIT.
    cutensorPlanPreference_t create_plan_pref() {
      cutensorPlanPreference_t pref{};
      cutensor_error_check(cutensorCreatePlanPreference(get_handle(), &pref, CUTENSOR_ALGO_DEFAULT, CUTENSOR_JIT_MODE_NONE),
                           "cutensorCreatePlanPreference");
      return pref;
    }

    // Destroy a given plan preference.
    void destroy_plan_pref(cutensorPlanPreference_t pref) {
      cutensor_error_check(cutensorDestroyPlanPreference(pref), "cutensorDestroyPlanPreference");
    }

    // Create an execution plan from an operation descriptor, plan preference and workspace size limit.
    cutensorPlan_t create_plan(cutensorOperationDescriptor_t op_desc, cutensorPlanPreference_t pref, std::uint64_t workspace_limit = 0) {
      cutensorPlan_t plan{};
      cutensor_error_check(cutensorCreatePlan(get_handle(), &plan, op_desc, pref, workspace_limit), "cutensorCreatePlan");
      return plan;
    }

    // Destroy a given execution plan.
    void destroy_plan(cutensorPlan_t plan) { cutensor_error_check(cutensorDestroyPlan(plan), "cutensorDestroyPlan"); }

    // Destroy a given operation descriptor.
    void destroy_op_desc(cutensorOperationDescriptor_t op_desc) {
      cutensor_error_check(cutensorDestroyOperationDescriptor(op_desc), "cutensorDestroyOperationDescriptor");
    }

    // Estimate the workspace size for an operation.
    std::uint64_t estimate_workspace(cutensorOperationDescriptor_t op_desc, cutensorPlanPreference_t pref,
                                     cutensorWorksizePreference_t ws_pref = CUTENSOR_WORKSPACE_DEFAULT) {
      std::uint64_t size = 0;
      cutensor_error_check(cutensorEstimateWorkspaceSize(get_handle(), op_desc, pref, ws_pref, &size), "cutensorEstimateWorkspaceSize");
      return size;
    }

    // Helper function to call the cuTENSOR permutation routine: B = alpha * opA(A).
    template <typename T>
    void permute_impl(T alpha, const_tensor_view<T> A, std::string_view idx_A, tensor_view<T> B, std::string_view idx_B) {
      auto &handle = get_handle();

      // create tensor descriptors
      auto desc_A = create_tensor_desc(A);
      auto desc_B = create_tensor_desc(B);

      // convert index strings to mode arrays
      auto modes_A = to_modes(idx_A);
      auto modes_B = to_modes(idx_B);

      // create operation descriptor
      cutensorOperationDescriptor_t op_desc{};
      cutensor_error_check(cutensorCreatePermutation(handle, &op_desc, desc_A, modes_A.data(), to_cutensor_unary_op(A.op), desc_B, modes_B.data(),
                                                     cutensor_compute_type<T>()),
                           "cutensorCreatePermutation");

      // create plan preference, estimate workspace, and create plan
      auto pref     = create_plan_pref();
      auto ws_limit = estimate_workspace(op_desc, pref);
      auto plan     = create_plan(op_desc, pref, ws_limit);

      // execute permutation
      cutensor_error_check(cutensorPermute(handle, plan, &alpha, A.data, B.data, nullptr /*stream*/), "cutensorPermute");

      // synchronize
      cuda_device_sync(synchronize, "cutensorPermute");

      // cleanup
      destroy_plan(plan);
      destroy_plan_pref(pref);
      destroy_op_desc(op_desc);
      destroy_tensor_desc(desc_A);
      destroy_tensor_desc(desc_B);
    }

    // Helper function to call the cuTENSOR elementwise binary routine: D = op_AC(alpha * op_A(A), gamma * op_C(C)).
    // D must have the same descriptor (shape/strides) as C but may point to different memory.
    template <typename T>
    void elementwise_binary_impl(T alpha, const_tensor_view<T> A, std::string_view idx_A, T gamma, const_tensor_view<T> C, std::string_view idx_C,
                                 tensor_view<T> D, binary_op op_AC) {
      auto &handle = get_handle();

      // create tensor descriptors (D descriptor must match C in shape/modes)
      auto desc_A = create_tensor_desc(A);
      auto desc_C = create_tensor_desc(C);
      auto desc_D = create_tensor_desc(D);

      // convert index strings to mode arrays
      auto modes_A = to_modes(idx_A);
      auto modes_C = to_modes(idx_C);

      // create operation descriptor (D has same modes as C)
      cutensorOperationDescriptor_t op_desc{};
      cutensor_error_check(cutensorCreateElementwiseBinary(handle, &op_desc, desc_A, modes_A.data(), to_cutensor_unary_op(A.op), desc_C,
                                                           modes_C.data(), to_cutensor_unary_op(C.op), desc_D, modes_C.data(),
                                                           to_cutensor_binary_op(op_AC), cutensor_compute_type<T>()),
                           "cutensorCreateElementwiseBinary");

      // create plan preference, estimate workspace, and create plan
      auto pref     = create_plan_pref();
      auto ws_limit = estimate_workspace(op_desc, pref);
      auto plan     = create_plan(op_desc, pref, ws_limit);

      // execute elementwise binary
      cutensor_error_check(cutensorElementwiseBinaryExecute(handle, plan, &alpha, A.data, &gamma, C.data, D.data, nullptr /*stream*/),
                           "cutensorElementwiseBinaryExecute");

      // synchronize
      cuda_device_sync(synchronize, "cutensorElementwiseBinaryExecute");

      // cleanup
      destroy_plan(plan);
      destroy_plan_pref(pref);
      destroy_op_desc(op_desc);
      destroy_tensor_desc(desc_A);
      destroy_tensor_desc(desc_C);
      destroy_tensor_desc(desc_D);
    }

    // Helper function to call the cuTENSOR elementwise trinary routine: D = op_ABC(op_AB(alpha * op_A(A), beta * op_B(B)), gamma * op_C(C)).
    // D must have the same descriptor (shape/strides) as C but may point to different memory.
    template <typename T>
    void elementwise_trinary_impl(T alpha, const_tensor_view<T> A, std::string_view idx_A, T beta, const_tensor_view<T> B, std::string_view idx_B,
                                  T gamma, const_tensor_view<T> C, std::string_view idx_C, tensor_view<T> D, binary_op op_AB, binary_op op_ABC) {
      auto &handle = get_handle();

      // create tensor descriptors (D descriptor must match C in shape/modes)
      auto desc_A = create_tensor_desc(A);
      auto desc_B = create_tensor_desc(B);
      auto desc_C = create_tensor_desc(C);
      auto desc_D = create_tensor_desc(D);

      // convert index strings to mode arrays
      auto modes_A = to_modes(idx_A);
      auto modes_B = to_modes(idx_B);
      auto modes_C = to_modes(idx_C);

      // create operation descriptor (D has same modes as C)
      cutensorOperationDescriptor_t op_desc{};
      cutensor_error_check(cutensorCreateElementwiseTrinary(handle, &op_desc, desc_A, modes_A.data(), to_cutensor_unary_op(A.op), desc_B,
                                                            modes_B.data(), to_cutensor_unary_op(B.op), desc_C, modes_C.data(),
                                                            to_cutensor_unary_op(C.op), desc_D, modes_C.data(), to_cutensor_binary_op(op_AB),
                                                            to_cutensor_binary_op(op_ABC), cutensor_compute_type<T>()),
                           "cutensorCreateElementwiseTrinary");

      // create plan preference, estimate workspace, and create plan
      auto pref     = create_plan_pref();
      auto ws_limit = estimate_workspace(op_desc, pref);
      auto plan     = create_plan(op_desc, pref, ws_limit);

      // execute elementwise trinary
      cutensor_error_check(cutensorElementwiseTrinaryExecute(handle, plan, &alpha, A.data, &beta, B.data, &gamma, C.data, D.data, nullptr /*stream*/),
                           "cutensorElementwiseTrinaryExecute");

      // synchronize
      cuda_device_sync(synchronize, "cutensorElementwiseTrinaryExecute");

      // cleanup
      destroy_plan(plan);
      destroy_plan_pref(pref);
      destroy_op_desc(op_desc);
      destroy_tensor_desc(desc_A);
      destroy_tensor_desc(desc_B);
      destroy_tensor_desc(desc_C);
      destroy_tensor_desc(desc_D);
    }

    // Helper function to call the cuTENSOR reduction routine: D = alpha * opReduce(op_A(A)) + beta * op_C(C).
    // D must have the same descriptor (shape/strides) as C but may point to different memory.
    // The modes of C/D must be a subset of the modes of A. The modes in A but not in C are reduced.
    template <typename T>
    void reduce_impl(T alpha, const_tensor_view<T> A, std::string_view idx_A, T beta, const_tensor_view<T> C, std::string_view idx_C,
                     tensor_view<T> D, binary_op op_reduce) {
      auto &handle = get_handle();

      // create tensor descriptors (D descriptor must match C in shape/modes)
      auto desc_A = create_tensor_desc(A);
      auto desc_C = create_tensor_desc(C);
      auto desc_D = create_tensor_desc(D);

      // convert index strings to mode arrays
      auto modes_A = to_modes(idx_A);
      auto modes_C = to_modes(idx_C);

      // create operation descriptor (D has same modes as C)
      cutensorOperationDescriptor_t op_desc{};
      cutensor_error_check(cutensorCreateReduction(handle, &op_desc, desc_A, modes_A.data(), to_cutensor_unary_op(A.op), desc_C, modes_C.data(),
                                                   to_cutensor_unary_op(C.op), desc_D, modes_C.data(), to_cutensor_binary_op(op_reduce),
                                                   cutensor_compute_type<T>()),
                           "cutensorCreateReduction");

      // create plan preference, estimate workspace, and create plan
      auto pref     = create_plan_pref();
      auto ws_limit = estimate_workspace(op_desc, pref);
      auto plan     = create_plan(op_desc, pref, ws_limit);

      // query the actual required workspace size from the plan
      std::uint64_t ws_size = 0;
      cutensor_error_check(cutensorPlanGetAttribute(handle, plan, CUTENSOR_PLAN_REQUIRED_WORKSPACE, &ws_size, sizeof(ws_size)),
                           "cutensorPlanGetAttribute");

      // allocate workspace
      void *workspace = nullptr;
      if (ws_size > 0) { device_error_check(cudaMalloc(&workspace, ws_size), "cudaMalloc"); }

      // execute reduction
      cutensor_error_check(cutensorReduce(handle, plan, &alpha, A.data, &beta, C.data, D.data, workspace, ws_size, nullptr /*stream*/),
                           "cutensorReduce");

      // synchronize
      cuda_device_sync(synchronize, "cutensorReduce");

      // free workspace
      if (workspace) { device_error_check(cudaFree(workspace), "cudaFree"); }

      // cleanup
      destroy_plan(plan);
      destroy_plan_pref(pref);
      destroy_op_desc(op_desc);
      destroy_tensor_desc(desc_A);
      destroy_tensor_desc(desc_C);
      destroy_tensor_desc(desc_D);
    }

    // Helper function to call the cuTENSOR contraction routine: D = alpha * op_A(A) * op_B(B) + beta * op_C(C).
    // D must have the same descriptor (shape/strides) as C but may point to different memory.
    template <typename T>
    void contract_impl(T alpha, const_tensor_view<T> A, std::string_view idx_A, const_tensor_view<T> B, std::string_view idx_B, T beta,
                       const_tensor_view<T> C, std::string_view idx_C, tensor_view<T> D) {
      auto &handle = get_handle();

      // create tensor descriptors (D descriptor must match C in shape/modes)
      auto desc_A = create_tensor_desc(A);
      auto desc_B = create_tensor_desc(B);
      auto desc_C = create_tensor_desc(C);
      auto desc_D = create_tensor_desc(D);

      // convert index strings to mode arrays
      auto modes_A = to_modes(idx_A);
      auto modes_B = to_modes(idx_B);
      auto modes_C = to_modes(idx_C);

      // create operation descriptor (D has same modes as C)
      cutensorOperationDescriptor_t op_desc{};
      cutensor_error_check(cutensorCreateContraction(handle, &op_desc, desc_A, modes_A.data(), to_cutensor_unary_op(A.op), desc_B, modes_B.data(),
                                                     to_cutensor_unary_op(B.op), desc_C, modes_C.data(), to_cutensor_unary_op(C.op), desc_D,
                                                     modes_C.data(), cutensor_compute_type<T>()),
                           "cutensorCreateContraction");

      // create plan preference, estimate workspace, and create plan
      auto pref     = create_plan_pref();
      auto ws_limit = estimate_workspace(op_desc, pref);
      auto plan     = create_plan(op_desc, pref, ws_limit);

      // query the actual required workspace size from the plan
      std::uint64_t ws_size = 0;
      cutensor_error_check(cutensorPlanGetAttribute(handle, plan, CUTENSOR_PLAN_REQUIRED_WORKSPACE, &ws_size, sizeof(ws_size)),
                           "cutensorPlanGetAttribute");

      // allocate workspace
      void *workspace = nullptr;
      if (ws_size > 0) { device_error_check(cudaMalloc(&workspace, ws_size), "cudaMalloc"); }

      // execute contraction
      cutensor_error_check(cutensorContract(handle, plan, &alpha, A.data, B.data, &beta, C.data, D.data, workspace, ws_size, nullptr /*stream*/),
                           "cutensorContract");

      // synchronize
      cuda_device_sync(synchronize, "cutensorContract");

      // free workspace
      if (workspace) { device_error_check(cudaFree(workspace), "cudaFree"); }

      // cleanup
      destroy_plan(plan);
      destroy_plan_pref(pref);
      destroy_op_desc(op_desc);
      destroy_tensor_desc(desc_A);
      destroy_tensor_desc(desc_B);
      destroy_tensor_desc(desc_C);
      destroy_tensor_desc(desc_D);
    }

  } // namespace

  // permute
  void permute(float alpha, const_tensor_view<float> A, std::string_view idx_A, tensor_view<float> B, std::string_view idx_B) {
    permute_impl(alpha, A, idx_A, B, idx_B);
  }
  void permute(double alpha, const_tensor_view<double> A, std::string_view idx_A, tensor_view<double> B, std::string_view idx_B) {
    permute_impl(alpha, A, idx_A, B, idx_B);
  }
  void permute(std::complex<float> alpha, const_tensor_view<std::complex<float>> A, std::string_view idx_A, tensor_view<std::complex<float>> B,
               std::string_view idx_B) {
    permute_impl(alpha, A, idx_A, B, idx_B);
  }
  void permute(std::complex<double> alpha, const_tensor_view<std::complex<double>> A, std::string_view idx_A, tensor_view<std::complex<double>> B,
               std::string_view idx_B) {
    permute_impl(alpha, A, idx_A, B, idx_B);
  }

  // elementwise_binary
  void elementwise_binary(float alpha, const_tensor_view<float> A, std::string_view idx_A, float gamma, const_tensor_view<float> C,
                          std::string_view idx_C, tensor_view<float> D, binary_op op_AC) {
    elementwise_binary_impl(alpha, A, idx_A, gamma, C, idx_C, D, op_AC);
  }
  void elementwise_binary(double alpha, const_tensor_view<double> A, std::string_view idx_A, double gamma, const_tensor_view<double> C,
                          std::string_view idx_C, tensor_view<double> D, binary_op op_AC) {
    elementwise_binary_impl(alpha, A, idx_A, gamma, C, idx_C, D, op_AC);
  }
  void elementwise_binary(std::complex<float> alpha, const_tensor_view<std::complex<float>> A, std::string_view idx_A, std::complex<float> gamma,
                          const_tensor_view<std::complex<float>> C, std::string_view idx_C, tensor_view<std::complex<float>> D, binary_op op_AC) {
    elementwise_binary_impl(alpha, A, idx_A, gamma, C, idx_C, D, op_AC);
  }
  void elementwise_binary(std::complex<double> alpha, const_tensor_view<std::complex<double>> A, std::string_view idx_A, std::complex<double> gamma,
                          const_tensor_view<std::complex<double>> C, std::string_view idx_C, tensor_view<std::complex<double>> D, binary_op op_AC) {
    elementwise_binary_impl(alpha, A, idx_A, gamma, C, idx_C, D, op_AC);
  }

  // elementwise_trinary
  void elementwise_trinary(float alpha, const_tensor_view<float> A, std::string_view idx_A, float beta, const_tensor_view<float> B,
                           std::string_view idx_B, float gamma, const_tensor_view<float> C, std::string_view idx_C, tensor_view<float> D,
                           binary_op op_AB, binary_op op_ABC) {
    elementwise_trinary_impl(alpha, A, idx_A, beta, B, idx_B, gamma, C, idx_C, D, op_AB, op_ABC);
  }
  void elementwise_trinary(double alpha, const_tensor_view<double> A, std::string_view idx_A, double beta, const_tensor_view<double> B,
                           std::string_view idx_B, double gamma, const_tensor_view<double> C, std::string_view idx_C, tensor_view<double> D,
                           binary_op op_AB, binary_op op_ABC) {
    elementwise_trinary_impl(alpha, A, idx_A, beta, B, idx_B, gamma, C, idx_C, D, op_AB, op_ABC);
  }
  void elementwise_trinary(std::complex<float> alpha, const_tensor_view<std::complex<float>> A, std::string_view idx_A, std::complex<float> beta,
                           const_tensor_view<std::complex<float>> B, std::string_view idx_B, std::complex<float> gamma,
                           const_tensor_view<std::complex<float>> C, std::string_view idx_C, tensor_view<std::complex<float>> D, binary_op op_AB,
                           binary_op op_ABC) {
    elementwise_trinary_impl(alpha, A, idx_A, beta, B, idx_B, gamma, C, idx_C, D, op_AB, op_ABC);
  }
  void elementwise_trinary(std::complex<double> alpha, const_tensor_view<std::complex<double>> A, std::string_view idx_A, std::complex<double> beta,
                           const_tensor_view<std::complex<double>> B, std::string_view idx_B, std::complex<double> gamma,
                           const_tensor_view<std::complex<double>> C, std::string_view idx_C, tensor_view<std::complex<double>> D, binary_op op_AB,
                           binary_op op_ABC) {
    elementwise_trinary_impl(alpha, A, idx_A, beta, B, idx_B, gamma, C, idx_C, D, op_AB, op_ABC);
  }

  // reduce
  void reduce(float alpha, const_tensor_view<float> A, std::string_view idx_A, float beta, const_tensor_view<float> C, std::string_view idx_C,
              tensor_view<float> D, binary_op op_reduce) {
    reduce_impl(alpha, A, idx_A, beta, C, idx_C, D, op_reduce);
  }
  void reduce(double alpha, const_tensor_view<double> A, std::string_view idx_A, double beta, const_tensor_view<double> C, std::string_view idx_C,
              tensor_view<double> D, binary_op op_reduce) {
    reduce_impl(alpha, A, idx_A, beta, C, idx_C, D, op_reduce);
  }
  void reduce(std::complex<float> alpha, const_tensor_view<std::complex<float>> A, std::string_view idx_A, std::complex<float> beta,
              const_tensor_view<std::complex<float>> C, std::string_view idx_C, tensor_view<std::complex<float>> D, binary_op op_reduce) {
    reduce_impl(alpha, A, idx_A, beta, C, idx_C, D, op_reduce);
  }
  void reduce(std::complex<double> alpha, const_tensor_view<std::complex<double>> A, std::string_view idx_A, std::complex<double> beta,
              const_tensor_view<std::complex<double>> C, std::string_view idx_C, tensor_view<std::complex<double>> D, binary_op op_reduce) {
    reduce_impl(alpha, A, idx_A, beta, C, idx_C, D, op_reduce);
  }

  // contract
  void contract(float alpha, const_tensor_view<float> A, std::string_view idx_A, const_tensor_view<float> B, std::string_view idx_B, float beta,
                const_tensor_view<float> C, std::string_view idx_C, tensor_view<float> D) {
    contract_impl(alpha, A, idx_A, B, idx_B, beta, C, idx_C, D);
  }
  void contract(double alpha, const_tensor_view<double> A, std::string_view idx_A, const_tensor_view<double> B, std::string_view idx_B, double beta,
                const_tensor_view<double> C, std::string_view idx_C, tensor_view<double> D) {
    contract_impl(alpha, A, idx_A, B, idx_B, beta, C, idx_C, D);
  }
  void contract(std::complex<float> alpha, const_tensor_view<std::complex<float>> A, std::string_view idx_A, const_tensor_view<std::complex<float>> B,
                std::string_view idx_B, std::complex<float> beta, const_tensor_view<std::complex<float>> C, std::string_view idx_C,
                tensor_view<std::complex<float>> D) {
    contract_impl(alpha, A, idx_A, B, idx_B, beta, C, idx_C, D);
  }
  void contract(std::complex<double> alpha, const_tensor_view<std::complex<double>> A, std::string_view idx_A,
                const_tensor_view<std::complex<double>> B, std::string_view idx_B, std::complex<double> beta,
                const_tensor_view<std::complex<double>> C, std::string_view idx_C, tensor_view<std::complex<double>> D) {
    contract_impl(alpha, A, idx_A, B, idx_B, beta, C, idx_C, D);
  }

} // namespace nda::tensor::device
