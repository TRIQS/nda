// Copyright (c) 2019--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides an iterator for nda::basic_array and nda::basic_array_view types.
 */

#pragma once

#include "./stdutil/array.hpp"

#include <array>
#include <cstddef>
#include <iterator>

namespace nda {

  /**
   * @addtogroup av_utils
   * @{
   */

  namespace detail {

    // N-dimensional rectangular grid iterator in C traversal order.
    template <int Rank>
    class grid_iterator {
      // Stride or number of elements to skip when increasing the iterator in the current dimension.
      long stri = 0;

      // Position of the iterator in the current dimension.
      long pos = 0;

      // Position times the stride in the current dimension.
      long offset = 0;

      // Iterator to the beginning of the Rank - 1 dimensional grid (excluding the current dimension).
      grid_iterator<Rank - 1> it_begin;

      // Iterator to the end of the Rank - 1 dimensional grid (excluding the current dimension).
      grid_iterator<Rank - 1> it_end;

      // Iterator to the Rank - 1 dimensional grid (excluding the current dimension).
      grid_iterator<Rank - 1> it;

      public:
      using iterator_category = std::forward_iterator_tag;
      using value_type        = long;
      using difference_type   = std::ptrdiff_t;
      using pointer           = long *;
      using reference         = long &;

      // Default constructor.
      grid_iterator() = default;

      // Construct an iterator from the shape of the grid, the stride of the grid and a flag indicating if the iterator
      // is at the end.
      grid_iterator(long const *lengths, long const *strides, bool at_end)
         : stri(strides[0]),
           pos(at_end ? lengths[0] : 0),
           offset(pos * stri),
           it_begin(lengths + 1, strides + 1, false),
           it_end(lengths + 1, strides + 1, true),
           it(it_begin) {}

      // Get the position/multi-dimensional index of the iterator.
      [[nodiscard]] std::array<long, Rank> indices() { return stdutil::front_append(it.indices(), pos); }

      // Dereference operator returns the sum of the offsets of every dimension = its linear index.
      [[nodiscard]] long operator*() const { return offset + *it; }

      // Member access operator returns the sum of the offsets of every dimension = its linear index.
      [[nodiscard]] long operator->() const { return operator*(); }

      // True if the positions of the iterators are equal in every dimension, false otherwise.
      [[nodiscard]] bool operator==(grid_iterator const &rhs) const { return ((rhs.pos == pos) and (rhs.it == it)); }

      // True if the positions of the iterators are not equal in every dimension, false otherwise.
      [[nodiscard]] bool operator!=(grid_iterator const &rhs) const { return not operator==(rhs); }

      // Prefix increment operator.
      grid_iterator &operator++() {
        // increment the iterator of the subgrid
        ++it;

        // if the iterator of the subgrid is at the end, reset it and increment the current position and offset
        if (it == it_end) { //FIXME [[unlikely]]
          ++pos;
          offset += stri;
          it = it_begin;
        }
        return *this;
      }

      // Postfix increment operator.
      grid_iterator operator++(int) {
        auto c = *this;
        ++(*this);
        return c;
      }
    };

    // Specialization of nda::grid_iterator for 1-dimensional grids.
    template <>
    class grid_iterator<1> {
      // Stride or number of elements to skip when increasing the iterator.
      long stri = 0;

      // Position of the iterator.
      long pos = 0;

      // Position times the stride.
      long offset = 0;

      public:
      using iterator_category = std::forward_iterator_tag;
      using value_type        = long;
      using difference_type   = std::ptrdiff_t;
      using pointer           = long *;
      using reference         = long &;

      // Default constructor.
      grid_iterator() = default;

      // Construct an iterator from the shape of the grid, the stride of the grid and a flag indicating if the iterator
      // is at the end.
      grid_iterator(long const *lengths, long const *strides, bool at_end) : stri(strides[0]), pos(at_end ? lengths[0] : 0), offset(pos * stri) {}

      // Get the position/index of the iterator.
      [[nodiscard]] std::array<long, 1> indices() { return {pos}; }

      // Dereference operator returns the offset = its linear index.
      [[nodiscard]] long operator*() const { return offset; }

      // Member access operator returns the offset = its linear index.
      [[nodiscard]] long operator->() const { return operator*(); }

      // True if the positions of the iterators are equal, false otherwise.
      [[nodiscard]] bool operator==(grid_iterator const &rhs) const { return (rhs.pos == pos); }

      // True if the positions of the iterators are not equal, false otherwise.
      [[nodiscard]] bool operator!=(grid_iterator const &rhs) const { return (rhs.pos != pos); }

      // Prefix increment operator increments the offset by the stride and the position by one.
      grid_iterator &operator++() {
        offset += stri;
        ++pos;
        return *this;
      }

      // Prefix decrement operator decrements the offset by the stride and the position by one.
      grid_iterator &operator--() {
        offset -= stri;
        --pos;
        return *this;
      }

      // Compound assignment addition operator increments the offset by n times the stride and the position by n.
      grid_iterator &operator+=(std::ptrdiff_t n) {
        offset += n * stri;
        pos += n;
        return *this;
      }

      // Binary addition of a grid iterator and an integer.
      [[nodiscard]] friend grid_iterator operator+(grid_iterator it, std::ptrdiff_t n) { return it += n; }

      // Binary subtraction of two grid iterators.
      [[nodiscard]] friend std::ptrdiff_t operator-(grid_iterator const &lhs, grid_iterator const &rhs) { return lhs.pos - rhs.pos; }

      // True if the position of the left hand side iterator is less than the position of the right hand side iterator,
      // false otherwise.
      [[nodiscard]] friend bool operator<(grid_iterator const &lhs, grid_iterator const &rhs) { return lhs.pos < rhs.pos; }

      // True if the position of the left hand side iterator is greater than the position of the right hand side
      // iterator, false otherwise.
      [[nodiscard]] friend bool operator>(grid_iterator const &lhs, grid_iterator const &rhs) { return lhs.pos > rhs.pos; }
    };

  } // namespace detail

  /**
   * @brief Iterator for nda::basic_array and nda::basic_array_view types.
   *
   * @details It is a <a href="https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator">LegacyRandomAccessIterator</a>
   * for 1-dimensional and a <a href="https://en.cppreference.com/w/cpp/named_req/ForwardIterator">LegacyForwardIterator</a>
   * for multi-dimensional arrays/views.
   *
   * Given the shape, strides and a pointer to the start of the data, the iterator uses an `nda::detail::grid_iterator`
   * to traverse the element in the array/view.
   *
   * @note The memory layout is always assumed to be C-style. For other layouts, one has to permute the given shape and
   * the strides according to the stride order.
   *
   * @tparam Rank Number of dimensions of the array.
   * @tparam T Type of the elements in the array (can be const).
   * @tparam Pointer Type of the pointer used to access the elements in the array (might be restricted depending on the
   * accessor).
   */
  template <int Rank, typename T, typename Pointer>
  class array_iterator {
    // Pointer to the data (to the first element).
    T *data = nullptr;

    // Shape of the array.
    std::array<long, Rank> len;

    // Strides of the array.
    std::array<long, Rank> stri;

    // Grid iterator.
    detail::grid_iterator<Rank> iter;

    public:
    /// Iterator category.
    using iterator_category = std::forward_iterator_tag;

    /// Value type.
    using value_type = T;

    /// Difference type.
    using difference_type = std::ptrdiff_t;

    /// Pointer type.
    using pointer = T *;

    /// Reference type.
    using reference = T &;

    /// Default constructor leaves the iterator in an uninitialized state.
    array_iterator() = default;

    /**
     * @brief Construct an iterator from the shape and the strides of an array/view, a pointer to its data and a flag
     * indicating if the iterator is at the end.
     *
     * @param lengths Shape of the array/view.
     * @param strides Strides of the array/view.
     * @param start Pointer to the data.
     * @param at_end Flag indicating if the iterator is at the end.
     */
    array_iterator(std::array<long, Rank> const &lengths, std::array<long, Rank> const &strides, T *start, bool at_end)
       : data(start), len(lengths), stri(strides), iter(len.data(), stri.data(), at_end) {}

    /**
     * @brief Get the current position/multi-dimensional index of the iterator.
     * @return `std::array<long, Rank>` containing the current position/multi-dimensional index of the iterator.
     */
    [[nodiscard]] auto indices() { return iter.indices(); }

    /**
     * @brief Dereference operator.
     * @return Reference to the element at the position of the iterator.
     */
    [[nodiscard]] value_type &operator*() const { return ((Pointer)data)[*iter]; }

    /**
     * @brief Member access operator.
     * @return Reference to the element at the position of the iterator.
     */
    [[nodiscard]] value_type &operator->() const { return operator*(); }

    /**
     * @brief Prefix increment operator.
     * @return Reference to the current iterator.
     */
    array_iterator &operator++() {
      ++iter;
      return *this;
    }

    /**
     * @brief Postfix increment operator.
     * @return Copy of the current iterator.
     */
    array_iterator operator++(int) {
      auto c = *this;
      ++iter;
      return c;
    }

    /**
     * @brief Equal-to operator.
     * @param rhs Other iterator to compare to.
     * @return True if the positions of the iterators are equal, false otherwise.
     */
    [[nodiscard]] bool operator==(array_iterator const &rhs) const { return (rhs.iter == iter); }

    /**
     * @brief Not-equal-to operator.
     * @param rhs Other iterator to compare to.
     * @return True if the positions of the iterators are not equal, false otherwise.
     */
    [[nodiscard]] bool operator!=(array_iterator const &rhs) const { return (!operator==(rhs)); }
  };

  /**
   * @brief Specialization of nda::array_iterator for 1-dimensional grids.
   *
   * @details It is a <a href="https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator">LegacyRandomAccessIterator</a>.
   *
   * @tparam T Type of the elements in the array (can be const).
   * @tparam Pointer Type of the pointer used to access the elements in the array (might be restricted
   * depending on the accessor).
   */
  template <typename T, typename Pointer>
  class array_iterator<1, T, Pointer> {
    // Pointer to the data.
    T *data = nullptr;

    // Shape of the array.
    std::array<long, 1> len{};

    // Strides of the array.
    std::array<long, 1> stri{};

    // Grid iterator.
    detail::grid_iterator<1> iter;

    public:
    /// Iterator category.
    using iterator_category = std::random_access_iterator_tag;

    /// Value type.
    using value_type = T;

    /// Difference type.
    using difference_type = std::ptrdiff_t;

    /// Pointer type.
    using pointer = T *;

    /// Reference type.
    using reference = T &;

    /// Default constructor leaves the iterator in an uninitialized state.
    array_iterator() = default;

    /**
     * @brief Construct an iterator from the shape and the strides of an array/view, a pointer to its data and a flag
     * indicating if the iterator is at the end.
     *
     * @param lengths Shape of the array/view.
     * @param strides Stride of the array/view.
     * @param start Pointer to the data.
     * @param at_end Flag indicating if the iterator is at the end.
     */
    array_iterator(std::array<long, 1> const &lengths, std::array<long, 1> const &strides, T *start, bool at_end)
       : data(start), len(lengths), stri(strides), iter(len.data(), stri.data(), at_end) {}

    /**
     * @brief Get the current position/index of the iterator.
     * @return `std::array<long, 1>` containing the current position/index of the iterator.
     */
    [[nodiscard]] auto indices() { return iter.indices(); }

    /**
     * @brief Dereference operator.
     * @return Reference to the element at the position of the iterator.
     */
    [[nodiscard]] T &operator*() const { return ((Pointer)data)[*iter]; }

    /**
     * @brief Member access operator.
     * @return Reference to the element at the position of the iterator.
     */
    T &operator->() const { return operator*(); }

    /**
     * @brief Prefix increment operator.
     * @return Reference to the current iterator.
     */
    array_iterator &operator++() {
      ++iter;
      return *this;
    }

    /**
     * @brief Postfix increment operator.
     * @return Copy of the current iterator.
     */
    array_iterator operator++(int) {
      auto c = *this;
      ++iter;
      return c;
    }

    /**
     * @brief Prefix decrement operator.
     * @return Reference to the current iterator.
     */
    array_iterator &operator--() {
      --iter;
      return *this;
    }

    /**
     * @brief Postfix decrement operator.
     * @return Copy of the current iterator.
     */
    array_iterator operator--(int) {
      auto c = *this;
      --iter;
      return c;
    }

    /**
     * @brief Equal-to operator.
     * @param rhs Right hand side operand.
     * @return True if the positions of the iterators are equal, false otherwise.
     */
    [[nodiscard]] bool operator==(array_iterator const &rhs) const { return (rhs.iter == iter); }

    /**
     * @brief Not-equal-to operator.
     * @param rhs Right hand side operand.
     * @return True if the positions of the iterators are not equal, false otherwise.
     */
    [[nodiscard]] bool operator!=(array_iterator const &rhs) const { return (!operator==(rhs)); }

    /**
     * @brief Compound assignment addition operator increments the iterator a given number of times.
     *
     * @param n Number of times to increment the iterator.
     * @return Reference to the current iterator.
     */
    array_iterator &operator+=(std::ptrdiff_t n) {
      iter += n;
      return *this;
    }

    /**
     * @brief Compound assignment subtraction operator decrements the iterator a given number of times.
     *
     * @param n Number of times to decrement the iterator.
     * @return Reference to the current iterator.
     */
    array_iterator &operator-=(std::ptrdiff_t n) {
      iter += (-n);
      return *this;
    }

    /**
     * @brief Binary addition of an integer with an 1-dimensional array iterator.
     *
     * @param n Integer.
     * @param it 1-dimensional array iterator.
     * @return Array iterator incremented n times.
     */
    [[nodiscard]] friend array_iterator operator+(std::ptrdiff_t n, array_iterator it) { return it += n; }

    /**
     * @brief Binary addition of an 1-dimensional array iterator with an integer.
     *
     * @param it 1-dimensional array iterator.
     * @param n Integer.
     * @return Array iterator incremented n times.
     */
    [[nodiscard]] friend array_iterator operator+(array_iterator it, std::ptrdiff_t n) { return it += n; }

    /**
     * @brief Binary subtraction of an 1-dimensional array iterator with an integer.
     *
     * @param it 1-dimensional array iterator.
     * @param n Integer.
     * @return Array iterator decremented n times.
     */
    [[nodiscard]] friend array_iterator operator-(array_iterator it, std::ptrdiff_t n) { return it -= n; }

    /**
     * @brief Binary subtraction of two 1-dimensional array iterators.
     *
     * @param lhs Left hand side operand.
     * @param rhs Right hand side operand.
     * @return Difference between their positions.
     */
    [[nodiscard]] friend std::ptrdiff_t operator-(array_iterator const &lhs, array_iterator const &rhs) { return lhs.iter - rhs.iter; }

    /**
     * @brief Subscript operator.
     *
     * @param n Number of times to increment the iterator before dereferencing it.
     * @return Reference to the element at the position of the incremented iterator.
     */
    [[nodiscard]] T &operator[](std::ptrdiff_t n) { return ((Pointer)data)[*(iter + n)]; }

    // FIXME C++20 ? with <=> operator
    /**
     * @brief Less-than comparison operator for two 1-dimensional array iterators.
     *
     * @param lhs Left hand side operand.
     * @param rhs Right hand side operand.
     * @return True if the position of the left hand side iterator is less than the position of the right hand side
     * iterator, false otherwise.
     */
    [[nodiscard]] friend bool operator<(array_iterator const &lhs, array_iterator const &rhs) { return lhs.iter < rhs.iter; }

    /**
     * @brief Greater-than comparison operator for two 1-dimensional array iterators.
     *
     * @param lhs Left hand side operand.
     * @param rhs Right hand side operand.
     * @return True if the position of the left hand side iterator is greater than the position of the right hand side
     * iterator, false otherwise.
     */
    [[nodiscard]] friend bool operator>(array_iterator const &lhs, array_iterator const &rhs) { return lhs.iter > rhs.iter; }

    /**
     * @brief Less-than or equal-to comparison operator for two 1-dimensional array iterators.
     *
     * @param lhs Left hand side operand.
     * @param rhs Right hand side operand.
     * @return True if the position of the left hand side iterator is less than or equal to the position of the right
     * hand side iterator, false otherwise.
     */
    [[nodiscard]] friend bool operator<=(array_iterator const &lhs, array_iterator const &rhs) { return not(lhs.iter > rhs.iter); }

    /**
     * @brief Greater-than or equal-to comparison operator for two 1-dimensional array iterators.
     *
     * @param lhs Left hand side operand.
     * @param rhs Right hand side operand.
     * @return True if the position of the left hand side iterator is greater than or equal to the position of the right
     * hand side iterator, false otherwise.
     */
    [[nodiscard]] friend bool operator>=(array_iterator const &lhs, array_iterator const &rhs) { return not(lhs.iter < rhs.iter); }
  };

  /** @} */

} // namespace nda
