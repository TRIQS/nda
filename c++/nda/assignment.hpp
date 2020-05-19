#pragma once
#include <cstring>
#include "./traits.hpp"

namespace nda {

  // Case where RHS has NdArray concept, but RHS is not a scalar (i.e. LHS:element_type, it could still be
  // an array in case of array of array
  template <typename LHS, typename RHS>
  void assign_from(LHS &lhs, RHS const &rhs) NDA_REQUIRES(is_ndarray_v<RHS> and not is_scalar_for_v<RHS, LHS>) {

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
  // RHS is a scalar for LHS (could be an array of array).

    // FIXME : PRIVATE 
    // GOES into the _impl_basic_array_view_common
  namespace details {

    // assign to a scalar, for array, no distinction for Matrix Algebra yet.
    // isolate this part which is reused in assign_from below
    template <typename LHS, typename RHS>
    void assign_from_scalar_array(LHS &lhs, RHS const &rhs) {
      // LHS is not a matrix, we simply do a multiple for... for loop on each dimension
      // we make a special implementation if the array is 1d strided or contiguous
      if constexpr (has_layout_strided_1d<LHS>) { // possibly contiguous
        const long L             = lhs.size();
        auto *__restrict const p = lhs.data_start(); // no alias possible here !
        if constexpr (has_layout_contiguous<LHS>) {
          for (long i = 0; i < L; ++i) p[i] = rhs;
        } else {
          const long s = lhs.indexmap().min_stride();
          for (long i = 0; i < L; i += s) p[i] = rhs;
        }
      } else {
        auto l = [&lhs, &rhs](auto const &... args) { lhs(args...) = rhs; };
        nda::for_each_static<LHS::idx_map_t::static_extents_encoded, LHS::idx_map_t::stride_order_encoded>(lhs.shape(), l);
      }
    }
  } // namespace details

  // -----------------------------------------------------
  // RHS is a scalar for LHS (could be an array of array).
  template <typename LHS, typename RHS>
  void assign_from(LHS &lhs, RHS const &rhs) NDA_REQUIRES(is_scalar_for_v<RHS, LHS>) {

    static_assert(is_regular_or_view_v<LHS>, "Internal error : LHS must a container");
    static_assert(!LHS::is_const, "Cannot assign to a const view !");

    if constexpr (get_algebra<LHS> != 'M') {
      details::assign_from_scalar_array(lhs, rhs);
    } else { // LHS is not a matrix. A scalar has to be interpreted as a unit matrix
      // FIXME : A priori faster to put 0 everywhere and then change the diag to avoid the if.
      // FIXME : Benchmark and confirm
      if constexpr (is_scalar_or_convertible_v<RHS>)
        details::assign_from_scalar_array(lhs, 0);
      else
        details::assign_from_scalar_array(lhs, RHS{0 * rhs}); //FIXME : improve this
      // on diagonal only
      const long imax = lhs.extent(0);
      for (long i = 0; i < imax; ++i) lhs(i, i) = rhs;
    }
  }

} // namespace nda
