// Copyright (c) 2024--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a C++ interface for various cuTENSOR routines.
 */

#pragma once

#include "../tools.hpp"

#include <complex>
#include <string_view>

namespace nda::tensor::device {

  void set_synchronization(bool do_sync) noexcept;
  bool get_synchronization() noexcept;

  void permute(float alpha, const_tensor_view<float> A, std::string_view idx_A, tensor_view<float> B, std::string_view idx_B);
  void permute(double alpha, const_tensor_view<double> A, std::string_view idx_A, tensor_view<double> B, std::string_view idx_B);
  void permute(std::complex<float> alpha, const_tensor_view<std::complex<float>> A, std::string_view idx_A, tensor_view<std::complex<float>> B,
               std::string_view idx_B);
  void permute(std::complex<double> alpha, const_tensor_view<std::complex<double>> A, std::string_view idx_A, tensor_view<std::complex<double>> B,
               std::string_view idx_B);

  void elementwise_binary(float alpha, const_tensor_view<float> A, std::string_view idx_A, float gamma, const_tensor_view<float> C,
                          std::string_view idx_C, tensor_view<float> D, binary_op op_AC = binary_op::SUM);
  void elementwise_binary(double alpha, const_tensor_view<double> A, std::string_view idx_A, double gamma, const_tensor_view<double> C,
                          std::string_view idx_C, tensor_view<double> D, binary_op op_AC = binary_op::SUM);
  void elementwise_binary(std::complex<float> alpha, const_tensor_view<std::complex<float>> A, std::string_view idx_A, std::complex<float> gamma,
                          const_tensor_view<std::complex<float>> C, std::string_view idx_C, tensor_view<std::complex<float>> D,
                          binary_op op_AC = binary_op::SUM);
  void elementwise_binary(std::complex<double> alpha, const_tensor_view<std::complex<double>> A, std::string_view idx_A, std::complex<double> gamma,
                          const_tensor_view<std::complex<double>> C, std::string_view idx_C, tensor_view<std::complex<double>> D,
                          binary_op op_AC = binary_op::SUM);

  void elementwise_trinary(float alpha, const_tensor_view<float> A, std::string_view idx_A, float beta, const_tensor_view<float> B,
                           std::string_view idx_B, float gamma, const_tensor_view<float> C, std::string_view idx_C, tensor_view<float> D,
                           binary_op op_AB = binary_op::SUM, binary_op op_ABC = binary_op::SUM);
  void elementwise_trinary(double alpha, const_tensor_view<double> A, std::string_view idx_A, double beta, const_tensor_view<double> B,
                           std::string_view idx_B, double gamma, const_tensor_view<double> C, std::string_view idx_C, tensor_view<double> D,
                           binary_op op_AB = binary_op::SUM, binary_op op_ABC = binary_op::SUM);
  void elementwise_trinary(std::complex<float> alpha, const_tensor_view<std::complex<float>> A, std::string_view idx_A, std::complex<float> beta,
                           const_tensor_view<std::complex<float>> B, std::string_view idx_B, std::complex<float> gamma,
                           const_tensor_view<std::complex<float>> C, std::string_view idx_C, tensor_view<std::complex<float>> D,
                           binary_op op_AB = binary_op::SUM, binary_op op_ABC = binary_op::SUM);
  void elementwise_trinary(std::complex<double> alpha, const_tensor_view<std::complex<double>> A, std::string_view idx_A, std::complex<double> beta,
                           const_tensor_view<std::complex<double>> B, std::string_view idx_B, std::complex<double> gamma,
                           const_tensor_view<std::complex<double>> C, std::string_view idx_C, tensor_view<std::complex<double>> D,
                           binary_op op_AB = binary_op::SUM, binary_op op_ABC = binary_op::SUM);

  void reduce(float alpha, const_tensor_view<float> A, std::string_view idx_A, float beta, const_tensor_view<float> C, std::string_view idx_C,
              tensor_view<float> D, binary_op op_reduce = binary_op::SUM);
  void reduce(double alpha, const_tensor_view<double> A, std::string_view idx_A, double beta, const_tensor_view<double> C, std::string_view idx_C,
              tensor_view<double> D, binary_op op_reduce = binary_op::SUM);
  void reduce(std::complex<float> alpha, const_tensor_view<std::complex<float>> A, std::string_view idx_A, std::complex<float> beta,
              const_tensor_view<std::complex<float>> C, std::string_view idx_C, tensor_view<std::complex<float>> D,
              binary_op op_reduce = binary_op::SUM);
  void reduce(std::complex<double> alpha, const_tensor_view<std::complex<double>> A, std::string_view idx_A, std::complex<double> beta,
              const_tensor_view<std::complex<double>> C, std::string_view idx_C, tensor_view<std::complex<double>> D,
              binary_op op_reduce = binary_op::SUM);

  void contract(float alpha, const_tensor_view<float> A, std::string_view idx_A, const_tensor_view<float> B, std::string_view idx_B, float beta,
                const_tensor_view<float> C, std::string_view idx_C, tensor_view<float> D);
  void contract(double alpha, const_tensor_view<double> A, std::string_view idx_A, const_tensor_view<double> B, std::string_view idx_B, double beta,
                const_tensor_view<double> C, std::string_view idx_C, tensor_view<double> D);
  void contract(std::complex<float> alpha, const_tensor_view<std::complex<float>> A, std::string_view idx_A, const_tensor_view<std::complex<float>> B,
                std::string_view idx_B, std::complex<float> beta, const_tensor_view<std::complex<float>> C, std::string_view idx_C,
                tensor_view<std::complex<float>> D);
  void contract(std::complex<double> alpha, const_tensor_view<std::complex<double>> A, std::string_view idx_A,
                const_tensor_view<std::complex<double>> B, std::string_view idx_B, std::complex<double> beta,
                const_tensor_view<std::complex<double>> C, std::string_view idx_C, tensor_view<std::complex<double>> D);

} // namespace nda::tensor::device
