// Copyright (c) 2024--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a C++ interface for various TBLIS tensor routines.
 */

#pragma once

#include "../tools.hpp"

#include <complex>
#include <string_view>

namespace nda::tensor::tblis {

  void set(float alpha, tensor_view<float> A, std::string_view idx_A);
  void set(double alpha, tensor_view<double> A, std::string_view idx_A);
  void set(std::complex<float> alpha, tensor_view<std::complex<float>> A, std::string_view idx_A);
  void set(std::complex<double> alpha, tensor_view<std::complex<double>> A, std::string_view idx_A);

  void scale(float alpha, tensor_view<float> A, std::string_view idx_A);
  void scale(double alpha, tensor_view<double> A, std::string_view idx_A);
  void scale(std::complex<float> alpha, tensor_view<std::complex<float>> A, std::string_view idx_A);
  void scale(std::complex<double> alpha, tensor_view<std::complex<double>> A, std::string_view idx_A);

  float reduce(binary_op op, const_tensor_view<float> A, std::string_view idx_A);
  double reduce(binary_op op, const_tensor_view<double> A, std::string_view idx_A);
  std::complex<float> reduce(binary_op op, const_tensor_view<std::complex<float>> A, std::string_view idx_A);
  std::complex<double> reduce(binary_op op, const_tensor_view<std::complex<double>> A, std::string_view idx_A);

  float dot(const_tensor_view<float> A, std::string_view idx_A, const_tensor_view<float> B, std::string_view idx_B);
  double dot(const_tensor_view<double> A, std::string_view idx_A, const_tensor_view<double> B, std::string_view idx_B);
  std::complex<float> dot(const_tensor_view<std::complex<float>> A, std::string_view idx_A, const_tensor_view<std::complex<float>> B,
                          std::string_view idx_B);
  std::complex<double> dot(const_tensor_view<std::complex<double>> A, std::string_view idx_A, const_tensor_view<std::complex<double>> B,
                           std::string_view idx_B);

  void add(float alpha, const_tensor_view<float> A, std::string_view idx_A, float beta, tensor_view<float> B, std::string_view idx_B);
  void add(double alpha, const_tensor_view<double> A, std::string_view idx_A, double beta, tensor_view<double> B, std::string_view idx_B);
  void add(std::complex<float> alpha, const_tensor_view<std::complex<float>> A, std::string_view idx_A, std::complex<float> beta,
           tensor_view<std::complex<float>> B, std::string_view idx_B);
  void add(std::complex<double> alpha, const_tensor_view<std::complex<double>> A, std::string_view idx_A, std::complex<double> beta,
           tensor_view<std::complex<double>> B, std::string_view idx_B);

  void mult(float alpha, const_tensor_view<float> A, std::string_view idx_A, const_tensor_view<float> B, std::string_view idx_B, float beta,
            tensor_view<float> C, std::string_view idx_C);
  void mult(double alpha, const_tensor_view<double> A, std::string_view idx_A, const_tensor_view<double> B, std::string_view idx_B, double beta,
            tensor_view<double> C, std::string_view idx_C);
  void mult(std::complex<float> alpha, const_tensor_view<std::complex<float>> A, std::string_view idx_A, const_tensor_view<std::complex<float>> B,
            std::string_view idx_B, std::complex<float> beta, tensor_view<std::complex<float>> C, std::string_view idx_C);
  void mult(std::complex<double> alpha, const_tensor_view<std::complex<double>> A, std::string_view idx_A, const_tensor_view<std::complex<double>> B,
            std::string_view idx_B, std::complex<double> beta, tensor_view<std::complex<double>> C, std::string_view idx_C);

} // namespace nda::tensor::tblis
