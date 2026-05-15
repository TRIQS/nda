// Copyright (c) 2024--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Implementation details for tensor/interface/tblis_interface.hpp.
 */

#include "./tblis_interface.hpp"
#include "../../exceptions.hpp"

#include <tblis/tblis.h>

#include <complex>

namespace nda::tensor::tblis {

  // Import TBLIS types and functions.
  using namespace ::tblis;

  namespace {

    // Helper function to call set routine.
    template <typename T>
    void set_impl(T alpha, tensor_view<T> A, std::string_view idx_A) {
      tblis_scalar s(alpha);
      tblis_tensor t(A.data, A.ndim, A.extents, A.strides);
      tblis_tensor_set(nullptr, nullptr, &s, &t, idx_A.data());
    }

    // Helper function to call scale routine.
    template <typename T>
    void scale_impl(T alpha, tensor_view<T> A, std::string_view idx_A) {
      tblis_tensor t(alpha, A.op == unary_op::CONJ, A.data, A.ndim, A.extents, A.strides);
      tblis_tensor_scale(nullptr, nullptr, &t, idx_A.data());
    }

    // Map our binary_op enum to TBLIS reduce_t.
    reduce_t to_tblis_reduce_op(binary_op op) {
      switch (op) {
        case binary_op::SUM: return REDUCE_SUM;
        case binary_op::SUM_ABS: return REDUCE_SUM_ABS;
        case binary_op::MAX: return REDUCE_MAX;
        case binary_op::MAX_ABS: return REDUCE_MAX_ABS;
        case binary_op::MIN: return REDUCE_MIN;
        case binary_op::MIN_ABS: return REDUCE_MIN_ABS;
        case binary_op::NORM_2: return REDUCE_NORM_2;
        default: NDA_RUNTIME_ERROR << "nda::tensor::tblis::reduce: nda::tensor::binary_op has no TBLIS equivalent";
      }
    }

    // Helper function to call reduce routine.
    template <typename T>
    T reduce_impl(binary_op op, const_tensor_view<T> A, std::string_view idx_A) {
      tblis_tensor t(T{1}, A.op == unary_op::CONJ, A.data, A.ndim, A.extents, A.strides);
      tblis_scalar result(T{});
      len_type idx = 0;
      tblis_tensor_reduce(nullptr, nullptr, to_tblis_reduce_op(op), &t, idx_A.data(), &result, &idx);
      return result.as<T>();
    }

    // Helper function to call dot routine.
    template <typename T>
    T dot_impl(const_tensor_view<T> A, std::string_view idx_A, const_tensor_view<T> B, std::string_view idx_B) {
      tblis_tensor tA(T{1}, A.op == unary_op::CONJ, A.data, A.ndim, A.extents, A.strides);
      tblis_tensor tB(T{1}, B.op == unary_op::CONJ, B.data, B.ndim, B.extents, B.strides);
      tblis_scalar result(T{});
      tblis_tensor_dot(nullptr, nullptr, &tA, idx_A.data(), &tB, idx_B.data(), &result);
      return result.as<T>();
    }

    // Helper function to call add routine.
    template <typename T>
    void add_impl(T alpha, const_tensor_view<T> A, std::string_view idx_A, T beta, tensor_view<T> B, std::string_view idx_B) {
      tblis_tensor tA(alpha, A.op == unary_op::CONJ, A.data, A.ndim, A.extents, A.strides);
      tblis_tensor tB(beta, B.op == unary_op::CONJ, B.data, B.ndim, B.extents, B.strides);
      tblis_tensor_add(nullptr, nullptr, &tA, idx_A.data(), &tB, idx_B.data());
    }

    // Helper function to call mult routine.
    template <typename T>
    void mult_impl(T alpha, const_tensor_view<T> A, std::string_view idx_A, const_tensor_view<T> B, std::string_view idx_B, T beta, tensor_view<T> C,
                   std::string_view idx_C) {
      tblis_tensor tA(alpha, A.op == unary_op::CONJ, A.data, A.ndim, A.extents, A.strides);
      tblis_tensor tB(T{1}, B.op == unary_op::CONJ, B.data, B.ndim, B.extents, B.strides);
      tblis_tensor tC(beta, C.op == unary_op::CONJ, C.data, C.ndim, C.extents, C.strides);
      tblis_tensor_mult(nullptr, nullptr, &tA, idx_A.data(), &tB, idx_B.data(), &tC, idx_C.data());
    }

  } // namespace

  // set
  void set(float alpha, tensor_view<float> A, std::string_view idx_A) { set_impl(alpha, A, idx_A); }
  void set(double alpha, tensor_view<double> A, std::string_view idx_A) { set_impl(alpha, A, idx_A); }
  void set(std::complex<float> alpha, tensor_view<std::complex<float>> A, std::string_view idx_A) { set_impl(alpha, A, idx_A); }
  void set(std::complex<double> alpha, tensor_view<std::complex<double>> A, std::string_view idx_A) { set_impl(alpha, A, idx_A); }

  // scale
  void scale(float alpha, tensor_view<float> A, std::string_view idx_A) { scale_impl(alpha, A, idx_A); }
  void scale(double alpha, tensor_view<double> A, std::string_view idx_A) { scale_impl(alpha, A, idx_A); }
  void scale(std::complex<float> alpha, tensor_view<std::complex<float>> A, std::string_view idx_A) { scale_impl(alpha, A, idx_A); }
  void scale(std::complex<double> alpha, tensor_view<std::complex<double>> A, std::string_view idx_A) { scale_impl(alpha, A, idx_A); }

  // reduce
  float reduce(binary_op op, const_tensor_view<float> A, std::string_view idx_A) { return reduce_impl(op, A, idx_A); }
  double reduce(binary_op op, const_tensor_view<double> A, std::string_view idx_A) { return reduce_impl(op, A, idx_A); }
  std::complex<float> reduce(binary_op op, const_tensor_view<std::complex<float>> A, std::string_view idx_A) { return reduce_impl(op, A, idx_A); }
  std::complex<double> reduce(binary_op op, const_tensor_view<std::complex<double>> A, std::string_view idx_A) { return reduce_impl(op, A, idx_A); }

  // dot
  float dot(const_tensor_view<float> A, std::string_view idx_A, const_tensor_view<float> B, std::string_view idx_B) {
    return dot_impl(A, idx_A, B, idx_B);
  }
  double dot(const_tensor_view<double> A, std::string_view idx_A, const_tensor_view<double> B, std::string_view idx_B) {
    return dot_impl(A, idx_A, B, idx_B);
  }
  std::complex<float> dot(const_tensor_view<std::complex<float>> A, std::string_view idx_A, const_tensor_view<std::complex<float>> B,
                          std::string_view idx_B) {
    return dot_impl(A, idx_A, B, idx_B);
  }
  std::complex<double> dot(const_tensor_view<std::complex<double>> A, std::string_view idx_A, const_tensor_view<std::complex<double>> B,
                           std::string_view idx_B) {
    return dot_impl(A, idx_A, B, idx_B);
  }

  // add
  void add(float alpha, const_tensor_view<float> A, std::string_view idx_A, float beta, tensor_view<float> B, std::string_view idx_B) {
    add_impl(alpha, A, idx_A, beta, B, idx_B);
  }
  void add(double alpha, const_tensor_view<double> A, std::string_view idx_A, double beta, tensor_view<double> B, std::string_view idx_B) {
    add_impl(alpha, A, idx_A, beta, B, idx_B);
  }
  void add(std::complex<float> alpha, const_tensor_view<std::complex<float>> A, std::string_view idx_A, std::complex<float> beta,
           tensor_view<std::complex<float>> B, std::string_view idx_B) {
    add_impl(alpha, A, idx_A, beta, B, idx_B);
  }
  void add(std::complex<double> alpha, const_tensor_view<std::complex<double>> A, std::string_view idx_A, std::complex<double> beta,
           tensor_view<std::complex<double>> B, std::string_view idx_B) {
    add_impl(alpha, A, idx_A, beta, B, idx_B);
  }

  // mult
  void mult(float alpha, const_tensor_view<float> A, std::string_view idx_A, const_tensor_view<float> B, std::string_view idx_B, float beta,
            tensor_view<float> C, std::string_view idx_C) {
    mult_impl(alpha, A, idx_A, B, idx_B, beta, C, idx_C);
  }
  void mult(double alpha, const_tensor_view<double> A, std::string_view idx_A, const_tensor_view<double> B, std::string_view idx_B, double beta,
            tensor_view<double> C, std::string_view idx_C) {
    mult_impl(alpha, A, idx_A, B, idx_B, beta, C, idx_C);
  }
  void mult(std::complex<float> alpha, const_tensor_view<std::complex<float>> A, std::string_view idx_A, const_tensor_view<std::complex<float>> B,
            std::string_view idx_B, std::complex<float> beta, tensor_view<std::complex<float>> C, std::string_view idx_C) {
    mult_impl(alpha, A, idx_A, B, idx_B, beta, C, idx_C);
  }
  void mult(std::complex<double> alpha, const_tensor_view<std::complex<double>> A, std::string_view idx_A, const_tensor_view<std::complex<double>> B,
            std::string_view idx_B, std::complex<double> beta, tensor_view<std::complex<double>> C, std::string_view idx_C) {
    mult_impl(alpha, A, idx_A, B, idx_B, beta, C, idx_C);
  }

} // namespace nda::tensor::tblis
