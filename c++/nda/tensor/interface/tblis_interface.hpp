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

#include "nda/concepts.hpp"
#include "tblis/tblis.h"

#include <string>
#include <vector>

namespace nda::tensor::nda_tblis {

// Following design choices of correaa@boost::multi 
template<class T> auto init_scalar = std::enable_if_t<sizeof(T*)==0>{};
template<> auto init_scalar<float               > = ::tblis::tblis_init_scalar_s;
template<> auto init_scalar<double              > = ::tblis::tblis_init_scalar_d;
template<> auto init_scalar<std::complex<float >> = ::tblis::tblis_init_scalar_c;
template<> auto init_scalar<std::complex<double>> = ::tblis::tblis_init_scalar_z;

template<class T> auto init_tensor = std::enable_if_t<sizeof(T*)==0>{};
template<> auto init_tensor<float               > = ::tblis::tblis_init_tensor_s;
template<> auto init_tensor<double              > = ::tblis::tblis_init_tensor_d;
template<> auto init_tensor<std::complex<float >> = ::tblis::tblis_init_tensor_c;
template<> auto init_tensor<std::complex<double>> = ::tblis::tblis_init_tensor_z;

template<class T> auto init_tensor_scaled = std::enable_if_t<sizeof(T*)==0>{};
template<> auto init_tensor_scaled<float               > = ::tblis::tblis_init_tensor_scaled_s;
template<> auto init_tensor_scaled<double              > = ::tblis::tblis_init_tensor_scaled_d;
template<> auto init_tensor_scaled<std::complex<float >> = ::tblis::tblis_init_tensor_scaled_c;
template<> auto init_tensor_scaled<std::complex<double>> = ::tblis::tblis_init_tensor_scaled_z;

template<class ValueType>
struct scalar : ::tblis::tblis_scalar {
  using value_type = ValueType;
  scalar() { init_scalar<std::decay_t<ValueType>>(this, 0); }
  scalar(ValueType v) { init_scalar<std::decay_t<ValueType>>(this, v); }
  scalar(scalar const&) = delete;
  scalar(scalar&& other) { init_scalar<std::decay_t<ValueType>>(this, ValueType(other.value())); }
  ValueType value() const{return ::tblis::tblis_scalar::get<ValueType>();}
};

template<class ValueType, int Rank>
struct tensor : ::tblis::tblis_tensor {

  using value_type = ValueType;
  static constexpr int rank = Rank;

  // since tblis types might not be consistent with nda
  std::array<::tblis::len_type   , rank> lens_;
  std::array<::tblis::stride_type, rank> strides_;

  explicit tensor(nda::MemoryArrayOfRank<Rank> auto&& a) :
    lens_(a.shape()), 
    strides_(a.strides()) { 
    init_tensor<std::decay_t<ValueType>>(this, rank, 
					 lens_.data(), 
				         const_cast<std::decay_t<ValueType>*>(a.data()), 
				         strides_.data());
  }
  explicit tensor(nda::MemoryArrayOfRank<Rank> auto&& a, ValueType val) :
    lens_(a.shape()),
    strides_(a.strides()) { 
    init_tensor_scaled<std::decay_t<ValueType>>(this, val, rank, 
                                         lens_.data(), 
                                         const_cast<std::decay_t<ValueType>*>(a.data()),
                                         strides_.data());
  }
  tensor(tensor const&) = delete;
  tensor(tensor&& other) : lens_{other.lens_}, strides_{other.strides_}{
    init_tensor_scaled<std::decay_t<ValueType>>(this, ValueType(other.scalar()), rank, 
					 lens_.data(), 
				         const_cast<std::decay_t<ValueType>*>(other.data()), 
				         strides_.data());
  }
  ValueType* data() const{return static_cast<ValueType*>(::tblis::tblis_tensor::data);}
//  ValueType scalar() const{return ::tblis::tblis_tensor::scalar.get<ValueType>();}
};

} // namespace nda::tensor::nda_tblis
