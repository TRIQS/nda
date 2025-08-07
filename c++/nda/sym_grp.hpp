// Copyright (c) 2023--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides tools to use symmetries with nda objects.
 */

#pragma once

#include "./nda.hpp"
#ifdef NDA_HAVE_MPI
#include "./mpi.hpp"
#endif

#include <itertools/omp_chunk.hpp>

#include <array>
#include <concepts>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <vector>

namespace nda {

  /**
   * @addtogroup av_sym
   * @{
   */

  /**
   * @brief A structure to capture combinations of complex conjugation and sign flip operations.
   */
  struct operation {
    /// Boolean value indicating a sign flip operation.
    bool sgn = false;

    /// Boolean value indicating a complex conjugation operation.
    bool cc = false;

    /**
     * @brief Multiplication operator for two operations.
     *
     * @details The sign flip (complex conjugation) operation is set to true in the resulting product iff one of the two
     * (exclusive or!) input operations has the sign flip (complex conjugation) operation set to true.
     *
     * @param rhs Right hand side operation.
     * @return The resulting operation.
     */
    operation operator*(operation const &rhs) { return operation{bool(sgn xor rhs.sgn), bool(cc xor rhs.cc)}; }

    /**
     * @brief Function call operator to apply the operation to a value.
     *
     * @tparam T Value type.
     * @param x Value to which the operation is applied.
     * @return The value after the operation has been applied.
     */
    template <typename T>
    T operator()(T const &x) const {
      if (sgn) return cc ? -conj(x) : -x;
      return cc ? conj(x) : x;
    }
  };

  /**
   * @brief Check if a multi-dimensional index is valid, i.e. not out of bounds, w.r.t. to a given nda::Array object.
   *
   * @tparam A nda::Array type.
   * @param a nda::Array object.
   * @param idx Multi-dimensional index.
   * @return True if the index is not out of bounds, false otherwise.
   */
  template <Array A>
  bool is_valid(A const &a, std::array<long, static_cast<std::size_t>(get_rank<A>)> const &idx) {
    for (auto i = 0; i < get_rank<A>; ++i) {
      if (not(0 <= idx[i] and idx[i] < a.shape()[i])) { return false; }
    }
    return true;
  }

  /**
   * @brief Concept defining a symmetry in nda.
   *
   * @details A symmetry consists of a callable type that can be called with a multi-dimensional index with the same
   * rank as a given array type and returns a tuple with a new multi-dimensional index and an nda::operation.
   *
   * The returned index corresponds to an element which is related to the element at the input index by the symmetry.
   *
   * The returned operation describes how the values of the elements are related, i.e. either via a sign flip, a complex
   * conjugation, or both.
   *
   * @tparam F Callable type.
   * @tparam A nda::Array type.
   * @tparam Idx Multi-dimensional index type.
   */
  template <typename F, typename A, typename Idx = std::array<long, static_cast<std::size_t>(get_rank<A>)>>
  concept NdaSymmetry = Array<A> and requires(F f, Idx const &idx) {
    { f(idx) } -> std::same_as<std::tuple<Idx, operation>>;
  };

  /**
   * @brief Concept defining an initializer function.
   *
   * @details An initializer function consists of a callable type that can be called with a multi-dimensional index with
   * the same rank as a given array type and returns a an object of the same type as the value type of the array.
   *
   * @tparam F Callable type.
   * @tparam A nda::Array type.
   * @tparam Idx Multi-dimensional index type.
   */
  template <typename F, typename A, typename Idx = std::array<long, static_cast<std::size_t>(get_rank<A>)>>
  concept NdaInitFunc = Array<A> and requires(F f, Idx const &idx) {
    { f(idx) } -> std::same_as<get_value_t<A>>;
  };

  /**
   * @brief Class representing a symmetry group.
   *
   * @details A symmetry group detects and stores all different symmetry classes associated with a given nda::Array
   * object.
   *
   * A symmetry class is simply a set of all the elements of the array that are related to each other by some symmetry.
   * The elements in a symmetry class have all the same values except for a possible sign flip or complex conjugation,
   * i.e. an nda::operation. The symmetry classes form a partition of all the elements of the array.
   *
   * @tparam F nda::NdaSymmetry type.
   * @tparam A nda::Array type.
   */
  template <typename F, typename A>
    requires(Array<A> && NdaSymmetry<F, A>)
  class sym_grp {
    public:
    /// Rank of the input array.
    static constexpr int ndims = get_rank<A>;

    /// Element type of a symmetry class.
    using sym_idx_t = std::pair<long, operation>;

    /// Symmetry class type.
    using sym_class_t = std::span<sym_idx_t>;

    /// Multi-dimensional index type.
    using arr_idx_t = std::array<long, static_cast<std::size_t>(ndims)>;

    private:
    // std::vector containing the different symmetry classes.
    std::vector<sym_class_t> sym_classes;

    // std::vector of the size of the input array to store the elements of the symmetry classes.
    std::vector<sym_idx_t> data;

    public:
    /**
     * @brief Get the symmetry classes.
     * @return std::vector containing the individual classes.
     */
    [[nodiscard]] std::vector<sym_class_t> const &get_sym_classes() const { return sym_classes; }

    /**
     * @brief Get the number of symmetry classes.
     * @return Number of detected symmetry classes.
     */
    [[nodiscard]] long num_classes() const { return sym_classes.size(); }

    /**
     * @brief Initialize an nda::Array using an nda::NdaInitFunc.
     *
     * @details The nda::NdaInitFunc object is evaluated only once per symmetry class. The result is then assigned to
     * all elements in the symmetry class after applying the nda::operation.
     *
     * @tparam H Callable type of nda::NdaInitFunc.
     * @param a nda::Array object to be initialized.
     * @param init_func Callable that is used to initialize the array.
     * @param parallel Parallelize using OpenMP, MPI, or both.
     */
    template <typename H>
      requires(NdaInitFunc<H, A>)
    void init(A &a, H const &init_func, bool parallel = false) const {
      auto init_with_sym = [&](sym_class_t const &sym_class) {
        auto idx           = a.indexmap().to_idx(sym_class[0].first);
        auto ref_val       = init_func(idx);
        std::apply(a, idx) = ref_val;
        for (auto const &[lin_idx, op] : sym_class) { std::apply(a, a.indexmap().to_idx(lin_idx)) = op(ref_val); }
      };

      if (parallel) {
#ifdef NDA_HAVE_OPENMP
#pragma omp parallel for
#endif // NDA_HAVE_OPENMP
#ifdef NDA_HAVE_MPI
        for (auto const &sym_class : mpi::chunk(sym_classes)) init_with_sym(sym_class);
        a = mpi::all_reduce(a);
#else
        for (auto const &sym_class : sym_classes) init_with_sym(sym_class);
#endif // NDA_HAVE_MPI
      } else {
        for (auto const &sym_class : sym_classes) init_with_sym(sym_class);
      }
    }

    /**
     * @brief Symmetrize an array and return the maximum symmetry violation and its corresponding array index.
     *
     * @note This actually requires the definition of an inverse operation but with the current implementation, all
     * operations are self-inverse (sign flip and complex conjugation).
     *
     * @param a nda::Array object to be symmetrized.
     * @return Maximum symmetry violation and corresponding array index.
    */
    std::pair<double, arr_idx_t> symmetrize(A &a) const {
      // loop over all symmetry classes
      double max_diff = 0.0;
      auto max_idx    = arr_idx_t{};
      for (auto const &sym_class : sym_classes) {
        // reference value for the symmetry class (arithmetic mean over all its elements)
        get_value_t<A> ref_val = 0.0;
        for (auto const &[lin_idx, op] : sym_class) { ref_val += op(std::apply(a, a.indexmap().to_idx(lin_idx))); }
        ref_val /= sym_class.size();

        // assign the reference value to all elements and calculate the violation
        for (auto const &[lin_idx, op] : sym_class) {
          auto mapped_val  = op(ref_val);
          auto mapped_idx  = a.indexmap().to_idx(lin_idx);
          auto current_val = std::apply(a, mapped_idx);
          auto diff        = std::abs(mapped_val - current_val);

          if (diff > max_diff) {
            max_diff = diff;
            max_idx  = mapped_idx;
          };

          std::apply(a, mapped_idx) = mapped_val;
        }
      }

      return std::pair{max_diff, max_idx};
    }

    /**
     * @brief Reduce an nda::Array to its representative data using symmetries.
     *
     * @param a nda::Array object.
     * @return std::vector of data values for the representative elements of each symmetry class.
     */
    [[nodiscard]] std::vector<get_value_t<A>> get_representative_data(A const &a) const {
      long len = sym_classes.size();
      auto vec = std::vector<get_value_t<A>>(len);
      for (long i : range(len)) vec[i] = std::apply(a, a.indexmap().to_idx(sym_classes[i][0].first));
      return vec;
    }

    /**
     * @brief Initialize a multi-dimensional array from its representative data using symmetries.
     *
     * @param a nda::Array object to be initialized.
     * @param vec std::vector of data values for the representative elements of each symmetry class.
    */
    template <typename V>
    void init_from_representative_data(A &a, V const &vec) const {
      static_assert(std::is_same_v<const get_value_t<A> &, decltype(vec[0])>);
      for (long i : range(vec.size())) {
        for (auto const &[lin_idx, op] : sym_classes[i]) { std::apply(a, a.indexmap().to_idx(lin_idx)) = op(vec[i]); }
      }
    }

    /// Default constructor for a symmetry group.
    sym_grp() = default;

    /**
     * @brief Construct a symmetry group for a given array and a list of its symmetries.
     *
     * @details It uses nda::for_each to loop over all possible multi-dimensional indices of the given array and applies
     * the symmetries to each index to generate the different symmetry classes.
     *
     * @param a nda::Array object.
     * @param sym_list List of symmetries containing nda::NdaSymmetry objects.
     * @param max_length Maximum recursion depth for out-of-bounds projection. Default is 0.
     */
    sym_grp(A const &a, std::vector<F> const &sym_list, long const max_length = 0) {
      // array to keep track of the indices already in a symmetry class
      array<bool, ndims> checked(a.shape());
      checked() = false;

      // initialize the data (we have as many elements as in the original array)
      data.reserve(a.size());

      // loop over all indices/elements
      for_each(checked.shape(), [&checked, &sym_list, max_length, this](auto... is) {
        if (not checked(is...)) {
          // if the index has not been checked yet, we start a new symmetry class
          auto class_start = data.end();

          // mark the current index as checked
          checked(is...) = true;

          // add it to the symmetry class as its representative together with the identity operation
          operation op;
          data.emplace_back(checked.indexmap()(is...), op);

          // apply all symmetries to the current index and generate the symmetry class
          auto idx        = std::array{is...};
          auto class_size = iterate(idx, op, checked, sym_list, max_length) + 1;

          // store the symmetry class
          sym_classes.emplace_back(class_start, class_size);
        }
      });
    }

    private:
    // Implementation of the actual symmetry reduction algorithm.
    long long iterate(std::array<long, static_cast<std::size_t>(get_rank<A>)> const &idx, operation const &op, array<bool, ndims> &checked,
                      std::vector<F> const &sym_list, long const max_length, long excursion_length = 0) {
      // count the number of new indices found by recursively applying the symmetries to the current index and to the
      // newly found indices
      long long segment_length = 0;

      // loop over all symmetries
      for (auto const &sym : sym_list) {
        // apply the symmetry to the current index
        auto [idxp, opp] = sym(idx);
        opp              = opp * op;

        // check if the new index is valid
        if (is_valid(checked, idxp)) {
          // if the new index is valid, check if it has been used already
          if (not std::apply(checked, idxp)) {
            // if it has not been used, mark it as checked
            std::apply(checked, idxp) = true;

            // add it to the symmetry class
            data.emplace_back(std::apply(checked.indexmap(), idxp), opp);

            // increment the segment length for the current index and recursively call the function with the new index
            // and the excursion length reset to zero
            segment_length += iterate(idxp, opp, checked, sym_list, max_length) + 1;
          }
        } else if (excursion_length < max_length) {
          // if the index is invalid, recursively call the function with the new index and the excursion length
          // incremented by one (the segment length is not incremented)
          segment_length += iterate(idxp, opp, checked, sym_list, max_length, ++excursion_length);
        }
      }

      // return the final value of the local segment length, which will be added to the segments length higher up in the
      // recursive call tree
      return segment_length;
    }
  };

  /** @} */

} // namespace nda
