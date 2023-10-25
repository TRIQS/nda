// Copyright (c) 2023 Simons Foundation
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
// Authors: Dominik Kiese

#pragma once
#include "nda.hpp"
#include "mpi.hpp"
#include <itertools/omp_chunk.hpp>

namespace nda {
  /**
    * A structure to capture combinations of complex conjugation and sign flip.
    */
  struct operation {
    /*@{*/
    bool sgn = false; /**< change sign? */
    bool cc  = false; /**< complex conjugation? */
    /*@}*/

    operation operator*(operation const &x) { return operation{bool(sgn xor x.sgn), bool(cc xor x.cc)}; }

    template <typename T>
    T operator()(T const &x) const {
      if (sgn) return cc ? -conj(x) : -x;
      return cc ? conj(x) : x;
    }
  };

  template <Array A>
  bool is_valid(A const &x, std::array<long, static_cast<std::size_t>(get_rank<A>)> const &idx) {

    for (auto i = 0; i < get_rank<A>; ++i) {
      if (not(0 <= idx[i] && idx[i] < x.shape()[i])) { return false; }
    }

    return true;
  }

  /**
   * Symmetry concept: NdaSymmetry accepts an array index and returns new array index & operation
   * @tparam F Anything callable with idx_t
   * @tparam A Anything modeling NdArray
   * @tparam idx_t Index for NdArray
   */
  template <typename F, typename A, typename idx_t = std::array<long, static_cast<std::size_t>(get_rank<A>)>>
  concept NdaSymmetry = Array<A> and //
     requires(F f, idx_t const &idx) {
       { f(idx) } -> std::same_as<std::tuple<idx_t, operation>>;
     };

  /**
   * Init function concept: NdaInitFunc accepts an array index and returns array value type
   * @tparam F Anything callable with idx_t
   * @tparam A Anything modeling NdArray
   * @tparam idx_t Index for NdArray
   */
  template <typename F, typename A, typename idx_t = std::array<long, static_cast<std::size_t>(get_rank<A>)>>
  concept NdaInitFunc = Array<A> and //
     requires(F f, idx_t const &idx) {
       { f(idx) } -> std::same_as<get_value_t<A>>;
     };

  /**
   * The sym_grp class 
   * @tparam F Anything modeling NdaSymmetry with A
   * @tparam A Anything modeling NdArray
   */
  template <typename F, typename A>
    requires(Array<A> && NdaSymmetry<F, A>)
  class sym_grp {

    public:
    /*@{*/
    static constexpr int ndims = get_rank<A>;                                       /**< rank of input array */
    using sym_idx_t            = std::pair<long, operation>;                        /**< return type of F */
    using sym_class_t          = std::span<sym_idx_t>;                              /**< symmetry class type */
    using arr_idx_t            = std::array<long, static_cast<std::size_t>(ndims)>; /**< index type of A */
    /*@}*/

    private:
    std::vector<sym_class_t> sym_classes; // list of classes
    std::vector<sym_idx_t> data;          // long list of all elements (to allocate contigous block of memory)

    public:
    /**
     * Accessor for symmetry classes
     * @return Vector including the individual classes
    */
    [[nodiscard]] std::vector<sym_class_t> const &get_sym_classes() const { return sym_classes; }

    /**
     * Accessor for number of symmetry classes
     * @return Number of deduced symmetry classes
    */
    long num_classes() const { return sym_classes.size(); }

    /**
     * Initializer method: Iterates over all classes and propagates result from evaluation of init function
     * @tparam H Anything modeling NdaInitFunction with NdArray A
     * @param x An NdArray
     * @param init_func The init function to be used
     * @param parallel Switch to enable parallel evaluation of init_func. Default is false
    */
    template <typename H>
      requires(NdaInitFunc<H, A>)
    void init(A &x, H const &init_func, bool parallel = false) const {

      if (parallel) {
        // reset input array to allow for mpi reduction
        x() = 0.0;

        #pragma omp parallel
        for (auto const &sym_class : itertools::omp_chunk(mpi::chunk(sym_classes))) {
          auto idx           = x.indexmap().to_idx(sym_class[0].first);
          auto ref_val       = init_func(idx);
          std::apply(x, idx) = ref_val;
          for (auto const &[lin_idx, op] : sym_class) { std::apply(x, x.indexmap().to_idx(lin_idx)) = op(ref_val); }
        }

        // distribute data among all ranks
        x = mpi::all_reduce(x);

      } else {
        for (auto const &sym_class : sym_classes) {
          auto idx           = x.indexmap().to_idx(sym_class[0].first);
          auto ref_val       = init_func(idx);
          std::apply(x, idx) = ref_val;
          for (auto const &[lin_idx, op] : sym_class) { std::apply(x, x.indexmap().to_idx(lin_idx)) = op(ref_val); }
        }
      }
    }

    // NOTE: this actually requires the definition of an inverse operation, but with the current implementation
    //       operations are anyways self-inverse

    /**
     * Symmetrization method: Symmetrizes an array returning the maximum symmetry violation and its corresponding array index
     * @param x An NdArray
     * @return Maximum symmetry violation and corresponding array index
    */
    std::pair<double, arr_idx_t> symmetrize(A &x) const {

      double max_diff = 0.0;
      auto max_idx    = arr_idx_t{};

      for (auto const &sym_class : sym_classes) {
        get_value_t<A> ref_val = 0.0;

        for (auto const &[lin_idx, op] : sym_class) { ref_val += op(std::apply(x, x.indexmap().to_idx(lin_idx))); }

        ref_val /= sym_class.size();

        for (auto const &[lin_idx, op] : sym_class) {
          auto mapped_val  = op(ref_val);
          auto mapped_idx  = x.indexmap().to_idx(lin_idx);
          auto current_val = std::apply(x, mapped_idx);
          auto diff        = std::abs(mapped_val - current_val);

          if (diff > max_diff) {
            max_diff = diff;
            max_idx  = mapped_idx;
          };

          std::apply(x, mapped_idx) = mapped_val;
        }
      }

      return std::pair{max_diff, max_idx};
    }

    /**
     * Reduce multidimensional array to its representative data using symmetries
     * @param x An NdArray
     * @return Vector of data values for the representatives elements of each symmetry class
    */
    [[nodiscard]] std::vector<get_value_t<A>> get_representative_data(A const &x) const {
      long const len = sym_classes.size();
      std::vector<get_value_t<A>> vec(len);
      for (auto const i : range(len)) vec[i] = std::apply(x, x.indexmap().to_idx(sym_classes[i][0].first));
      return vec;
    }

    /**
     * Init multidimensional array from its representative data using symmetries
     * @param x An NdArray
     * @param vec Vector or vector view of data values for the representatives elements of each symmetry class
    */
    template <typename V>
    void init_from_representative_data(A &x, V const &vec) const {
      static_assert(std::is_same_v<const get_value_t<A> &, decltype(vec[0])>);
      for (auto const i : range(vec.size())) {
        auto const ref_val = vec[i];
        for (auto const &[lin_idx, op] : sym_classes[i]) { std::apply(x, x.indexmap().to_idx(lin_idx)) = op(ref_val); }
      }
    };

    /**
     * Default constructor for sym_grp class
    */
    sym_grp() = default;

    /**
     * Constructor for sym_grp class
     * @param x An NdArray
     * @param sym_list List of symmetries modeling the NdaSymmetry concept
     * @param max_length Maximum recursion depth for out-of-bounds projection. Default is 0.
    */
    sym_grp(A const &x, std::vector<F> const &sym_list, long const max_length = 0) {

      // array to check whether index has been sorted into a symmetry class already
      array<bool, ndims> checked(x.shape());
      checked() = false;

      // initialize data array (we have as many elements as in the original nda array)
      data.reserve(x.size());

      for_each(checked.shape(), [&checked, &sym_list, max_length, this](auto... i) {
        if (not checked(i...)) {
          operation op;
          checked(i...)    = true; // this index is now checked and defines a new symmetry class
          auto idx         = std::array{i...};
          auto class_start = data.end();                   // the class is added to the end of the data list
          data.emplace_back(checked.indexmap()(i...), op); // every class is initialized by one representative with op = identity

          // apply all symmetries to current index and generate the symmetry class
          auto class_size = iterate(idx, op, checked, sym_list, max_length) + 1;

          sym_classes.emplace_back(class_start, class_size);
        }
      });
    }

    private:
    // implementation of the actual symmetry reduction algorithm
    long long iterate(std::array<long, static_cast<std::size_t>(get_rank<A>)> const &idx, operation const &op, array<bool, ndims> &checked,
                      std::vector<F> const &sym_list, long const max_length, long excursion_length = 0) {

      // initialize the local segment_length to 0 (we have not advanced to a new member of the symmetry class so far)
      long long segment_length = 0;

      for (auto const &sym : sym_list) {
        // apply the symmetry
        auto [idxp, opp] = sym(idx);
        opp              = opp * op;

        // check if index is valid, reset excursion_length iff we start from valid & unchecked index
        if (is_valid(checked, idxp)) {

          // check if index has been used already
          if (not std::apply(checked, idxp)) {
            std::apply(checked, idxp) = true; // this index is now checked

            // add new member to symmetry class and increment the segment_length
            data.emplace_back(std::apply(checked.indexmap(), idxp), opp);
            segment_length += iterate(idxp, opp, checked, sym_list, max_length) + 1;
          }

          // if index is invalid, increment excursion length and keep going (segment_length is not incremented)
        } else if (excursion_length < max_length) {
          segment_length += iterate(idxp, opp, checked, sym_list, max_length, ++excursion_length);
        }
      }

      // return the final value of the local segment_length, which will be added
      // to the segment_length higher up in the recursive call tree
      return segment_length;
    }
  };
} // namespace nda