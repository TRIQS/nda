#pragma once
#include <cstring>
#include "./traits.hpp"

//https://godbolt.org/g/tkNLoE
// https://godbolt.org/g/SqrgBH

namespace nda {

  // implementation of the assignment operator for the container
  // Case where RHS has NdArray concept, but RHS is not a scalar (i.e. LHS:element_type, it could still be
  // an array in case of array of array
  template <typename LHS, typename RHS>
  void assign_from(LHS &lhs, RHS const &rhs) REQUIRES(is_ndarray_v<RHS> and not is_scalar_for_v<RHS, LHS>) {

    static_assert(is_regular_or_view_v<LHS>, "Internal error : LHS must a container");
    static_assert(!LHS::is_const, "Cannot assign to a const view !");

#ifdef NDA_DEBUG
    if (lhs.shape() != rhs.shape())
      NDA_RUNTIME_ERROR << "Size mismatch in = "
                        << " : LHS " << lhs << " \n RHS = " << rhs;
#endif

    // general case if RHS is not a scalar (can be isp, expression...)
    static_assert(std::is_assignable_v<typename LHS::value_t &, get_value_t<RHS>>, "Assignment impossible for the type of RHS into the type of LHS");

    // If LHS and RHS are both 1d strided order or contiguous, and have the same stride order
    // we can make a 1d loop
    if constexpr ((get_layout_info<LHS>.stride_order == get_layout_info<RHS>.stride_order) // same stride order and both contiguous ...
                  and has_layout_strided_1d<LHS> and has_layout_strided_1d<RHS>) {
                  //and has_layout_contiguous<LHS> and has_layout_contiguous<RHS>) {
      //  NDA_PRINT("Assignment : linear computation optimisation");
      long L = lhs.size();
      for (long i = 0; i < L; ++i) lhs(_linear_index_t{i}) = rhs(_linear_index_t{i});
      //

    } else {
      auto l = [&lhs, &rhs](auto const &... args) { lhs(args...) = rhs(args...); };
      nda::for_each(lhs.shape(), l);
    }
  }

  // -----------------------------------------------------
  // implementation of the assignment operator for the container
  // Case where RHS has NdArray concept, but RHS is not a scalar (i.e. LHS:element_type, it could still be
  // an array in case of array of array
  template <typename LHS, typename RHS>
  void assign_from(LHS &lhs, RHS const &rhs) REQUIRES(is_scalar_for_v<RHS, LHS>) {

    static_assert(is_regular_or_view_v<LHS>, "Internal error : LHS must a container");
    static_assert(!LHS::is_const, "Cannot assign to a const view !");

    // RHS is a scalar for LHS
    // if LHS is a matrix, the unit has a specific interpretation.

    if constexpr (get_algebra<LHS> == 'M') {
      // FIXME : Foreach on diagonal only !
      // WRITE THE LLOP !SAME in compound op !!
      auto l = [&lhs, &rhs](auto &&x1, auto &&x2) {
        if (x1 == x2)
          lhs(x1, x2) = rhs;
        else {
          if constexpr (is_scalar_or_convertible_v<RHS>)
            lhs(x1, x2) = 0;
          else
            lhs(x1, x2) = RHS{0 * rhs}; //FIXME : improve this
        }
      }; // end lambda l
      nda::for_each(lhs.shape(), l);
    } else { // LHS is not a matrix
      auto l = [&lhs, &rhs](auto const &... args) { lhs(args...) = rhs; };
      nda::for_each(lhs.shape(), l);
    }
  }

  // ===========================  compound assignment ===========================================================

  template <char OP, typename LHS, typename RHS>
  void compound_assign_from(LHS &lhs, RHS const &rhs) {

    static_assert(!LHS::is_const, "Cannot assign to a const view !");
    static_assert((!std::is_const<typename LHS::value_t>::value), "Assignment : The value type of the LHS is const and cannot be assigned to !");
    static_assert(
       (!((OP == 'M' || OP == 'D') and (get_algebra<LHS> == 'M') and (not is_scalar_for_v<RHS, LHS>))),
       "*= and /= operator for non scalar RHS are deleted for a type modeling MutableMatrix (e.g. matrix, matrix_view) matrix, because this is ambiguous");

    // general case if RHS is not a scalar (can be isp, expression...)
    if constexpr (not is_scalar_for_v<RHS, LHS>) {

      static_assert(is_ndarray_v<RHS>, "Error");

      auto l = [&lhs, &rhs](auto const &... args) {
        if constexpr (OP == 'A') { lhs(args...) += rhs(args...); }
        if constexpr (OP == 'S') { lhs(args...) -= rhs(args...); }
        if constexpr (OP == 'M') { lhs(args...) *= rhs(args...); }
        if constexpr (OP == 'D') { lhs(args...) /= rhs(args...); }
      };
      nda::for_each(lhs.shape(), l);
    }

    // RHS is a scalar for LHS
    else {
      // if LHS is a matrix, the unit has a specific interpretation.
      if constexpr ((get_algebra<LHS> == 'M') and (OP == 'A' || OP == 'S')) {
        if (lhs.shape()[0] != lhs.shape()[1]) NDA_RUNTIME_ERROR << "Adding a number to a matrix only works if the matrix is square !";
        long s = lhs.shape()[0];
        for (long i = 0; i < s; ++i) { // diagonal only
          if constexpr (OP == 'A') { lhs(i, i) += rhs; }
          if constexpr (OP == 'S') { lhs(i, i) -= rhs; }
        }
      } else { // LHS is not a matrix
        long s = lhs.size();
        for (long i = 0; i < s; ++i) {
          if constexpr (OP == 'A') { lhs.data_start()[i] += rhs; }
          if constexpr (OP == 'S') { lhs.data_start()[i] -= rhs; }
          if constexpr (OP == 'M') { lhs.data_start()[i] *= rhs; }
          if constexpr (OP == 'D') { lhs.data_start()[i] /= rhs; }
        }
      }
    }
  }

} // namespace nda
