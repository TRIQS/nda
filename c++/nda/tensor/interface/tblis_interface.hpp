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

#include <array>
#include <string>
#include <vector>

namespace nda::tensor::nda_tblis {

  template <class ValueType>
  struct scalar : ::tblis::tblis_scalar {
    using value_type = ValueType;

    scalar() : ::tblis::tblis_scalar(ValueType{}) {}
    explicit scalar(ValueType v) : ::tblis::tblis_scalar(v) {}
    scalar(scalar const &) = delete;
    scalar(scalar &&other) : ::tblis::tblis_scalar(other.template as<ValueType>()) {}
    ValueType value() const { return this->template as<ValueType>(); }
  };

  template <class ValueType, int Rank>
  struct tensor : ::tblis::tblis_tensor {

    using value_type          = ValueType;
    static constexpr int rank = Rank;

    // since tblis types might not be consistent with nda
    std::array<::tblis::len_type, rank> lens_;
    std::array<::tblis::stride_type, rank> strides_;

    explicit tensor(nda::MemoryArrayOfRank<Rank> auto &&a, ValueType val = ValueType{1}, bool conj = false)
       : ::tblis::tblis_tensor(), lens_(to_lens(a.shape())), strides_(to_strides(a.strides())) {
      configure_tensor(const_cast<std::decay_t<ValueType> *>(a.data()), val, conj);
    }

    tensor(tensor const &) = delete;
    tensor(tensor &&other) : ::tblis::tblis_tensor(), lens_{other.lens_}, strides_{other.strides_} {
      this->type = other.type;
      this->conj = other.conj;
      this->scalar.reset(other.scalar);
      this->::tblis::tblis_tensor::data = other.data;
      this->ndim                        = other.ndim;
      this->len                         = lens_.data();
      this->stride                      = strides_.data();
    }

    ValueType *data() const { return static_cast<ValueType *>(::tblis::tblis_tensor::data); }
    //  ValueType scalar() const{return ::tblis::tblis_tensor::scalar.get<ValueType>();}

    private:
    template <typename Shape>
    static std::array<::tblis::len_type, rank> to_lens(Shape const &shape) {
      std::array<::tblis::len_type, rank> result{};
      for (int i = 0; i < rank; ++i) result[i] = static_cast<::tblis::len_type>(shape[i]);
      return result;
    }

    template <typename Strides>
    static std::array<::tblis::stride_type, rank> to_strides(Strides const &strides) {
      std::array<::tblis::stride_type, rank> result{};
      for (int i = 0; i < rank; ++i) result[i] = static_cast<::tblis::stride_type>(strides[i]);
      return result;
    }

    void configure_tensor(std::decay_t<ValueType> *ptr, ValueType alpha, bool conjugate) {
      this->type = ::tblis::type_tag<std::decay_t<ValueType>>::value;
      this->conj = (conjugate ? 1 : 0 );
      this->scalar.reset(alpha);
      this->::tblis::tblis_tensor::data = ptr;
      this->ndim                        = rank;
      this->len                         = lens_.data();
      this->stride                      = strides_.data();
    }
  };

} // namespace nda::tensor::nda_tblis
