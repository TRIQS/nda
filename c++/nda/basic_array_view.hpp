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
// Authors: Miguel Morales, Olivier Parcollet, Nils Wentzell

#pragma once
#include <cstring>
#include <memory>
#include <ranges>
#include "clef.hpp"
#include "declarations.hpp"
#include "exceptions.hpp"
#include "concepts.hpp"
#include "iterators.hpp"
#include "layout/slice_static.hpp"

// The std::swap is WRONG for a view because of the copy/move semantics of view.
// Use swap instead (the correct one, found by ADL).
namespace std {
  template <typename V, int R, typename L, char A, typename AP, typename OP, typename V2, int R2, typename L2, char A2, typename AP2, typename OP2>
  void swap(nda::basic_array_view<V, R, L, A, AP, OP> &a, nda::basic_array_view<V2, R2, L2, A2, AP2, OP2> &b) =
     delete; // std::swap disabled for basic_array_view. Use nda::swap instead (or simply swap, found by ADL).
}

namespace nda {

  // forward for friend declaration
  template <MemoryArray A, typename NewLayoutType>
  auto map_layout_transform(A &&a, NewLayoutType const &new_layout);

  // -----------------------------------------------

  template <typename ValueType, int Rank, typename LayoutPolicy, char Algebra, typename AccessorPolicy, typename OwningPolicy>
  class basic_array_view {

    static_assert((Algebra != 'N'), " Internal error : Algebra 'N' not supported");
    static_assert((Algebra != 'M') or (Rank == 2), " Internal error : Algebra 'A' only makes sense for rank 2");
    static_assert((Algebra != 'V') or (Rank == 1), " Internal error : Algebra 'V' only makes sense for rank 1");

    public:
    /// Type of the array's values
    using value_type = ValueType;
    /// Type of the memory layout policy
    using layout_policy_t = LayoutPolicy;
    /// Type of the memory layout
    using layout_t = typename LayoutPolicy::template mapping<Rank>;
    /// Type of the accessor policy
    using accessor_policy_t = AccessorPolicy;
    /// Type of the owning policy
    using owning_policy_t = OwningPolicy;
    /// Type of the memory handle
    using storage_t = typename OwningPolicy::template handle<ValueType>;
    /// The associated regular type
    using regular_type = basic_array<std::remove_const_t<ValueType>, Rank, C_layout, Algebra, heap<mem::get_addr_space<storage_t>>>;
    /// The number of dimensions of the array
    static constexpr int rank = Rank;

    private:
    // details for the common code with array
    using self_t                   = basic_array_view;
    static constexpr bool is_view  = true;
    static constexpr bool is_const = std::is_const_v<ValueType>;

    layout_t lay;
    storage_t sto;

    template <typename T, int R, typename L, char A, typename CP>
    friend class basic_array;

    template <typename T, int R, typename L, char A, typename AP, typename OP>
    friend class basic_array_view;

    template <typename L>
    static constexpr bool requires_runtime_check = not layout_property_compatible(L::template mapping<Rank>::layout_prop, layout_t::layout_prop);

    public:
    // FIXME : TRIQS PORTING
    // private constructor for the previous friend
    basic_array_view(layout_t const &idxm, storage_t st) : lay(idxm), sto(std::move(st)) {}

    public:
    // backward : FIXME : temporary to be removed
    [[deprecated]] basic_array_view<ValueType, Rank, LayoutPolicy, 'A', AccessorPolicy, OwningPolicy> as_array_view() { return {*this}; };
    [[deprecated]] basic_array_view<const ValueType, Rank, LayoutPolicy, 'A', AccessorPolicy, OwningPolicy> as_array_view() const { return {*this}; };

    // ------------------------------- constructors --------------------------------------------

    /// Construct an empty view.
    basic_array_view() = default;

    ///
    basic_array_view(basic_array_view &&) = default;

    /// Shallow copy. It copies the *view*, not the data.
    basic_array_view(basic_array_view const &) = default;

    /// Generic constructor from any MemoryArray type
    template <MemoryArrayOfRank<Rank> A>
      requires(
                 // Require same stride-order
                 (get_layout_info<A>.stride_order == layout_t::stride_order_encoded) and
                 // Require same underlying value_type
                 (std::is_same_v<std::remove_const_t<ValueType>, get_value_t<A>>) and
                 // Make sure that we have a const ValueType if our argument does
                 (std::is_const_v<ValueType> or !std::is_const_v<typename std::decay_t<A>::value_type>))
    // Explicit iff layout properties maybe be incompatible
    explicit(requires_runtime_check<typename std::decay_t<A>::layout_policy_t>) basic_array_view(A &&a) noexcept
       : lay(a.indexmap()), sto(a.storage()) {}

    /** 
     * [Advanced] From a pointer to **contiguous data**, and a shape.
     * NB : no control on the dimensions given.  
     *
     * @param p Pointer to the data
     * @param shape Shape of the view (contiguous)
     */
    basic_array_view(std::array<long, Rank> const &shape, ValueType *p) noexcept : basic_array_view(layout_t{shape}, p) {}

    /** 
     * [Advanced] From a pointer to data, and an idx_map 
     * NB : no control obvious on the dimensions given.  
     *
     * @param p Pointer to the data 
     * @param idxm Index Map (view can be non contiguous). If the offset is non zero, the view starts at p + idxm.offset()
     */
    basic_array_view(layout_t const &idxm, ValueType *p) noexcept : lay(idxm), sto{p} {}
    //basic_array_view(idx_map<Rank, StrideOrder> const &idxm, ValueType *p) : lay(idxm), sto{p, size_t(idxm.size() + idxm.offset())} {}

    // Move assignment not defined : will use the copy = since view must copy data

    /**
     * Construct from a std::array reference
     *
     * @tparam N The size of the std::array
     * @param a The array
     */
    template <size_t N>
    explicit basic_array_view(std::array<ValueType, N> &a) noexcept
      requires(Rank == 1)
       : basic_array_view{{long(N)}, a.data()} {}

    /**
     * Construct from a std::array const reference
     *
     * @tparam N The size of the std::array
     * @param a The array
     */
    template <size_t N>
    explicit basic_array_view(std::array<std::remove_const_t<ValueType>, N> const &a) noexcept
      requires(Rank == 1 and std::is_const_v<ValueType>)
       : basic_array_view{{long(N)}, a.data()} {}

    /**
     * Construct from a 1D Contiguous Range
     *
     * @tparam R The contiguous Range type
     * @param r The contiguous Range
     */
    template <std::ranges::contiguous_range R>
    explicit basic_array_view(R &r) noexcept
      requires(Rank == 1 and not MemoryArray<R>
               and (std::is_same_v<std::ranges::range_value_t<R>, ValueType> or std::is_same_v<const std::ranges::range_value_t<R>, ValueType>))
       : basic_array_view{{long(std::ranges::size(r))}, std::to_address(std::begin(r))} {}

    // ------------------------------- assign --------------------------------------------

    /// Same as the general case
    /// [C++ oddity : this case must be explicitly coded too]
    basic_array_view &operator=(basic_array_view const &rhs) noexcept {
      assign_from_ndarray(rhs);
      return *this;
    }

    /**
     * Copies the content of rhs into the view.
     * Pseudo code : 
     *     for all i,j,k,l,... : this[i,j,k,l,...] = rhs(i,j,k,l,...)
     *
     * The dimension of RHS must be large enough or behaviour is undefined.
     * 
     * If NDA_BOUNDCHECK is defined, the bounds are checked.
     *
     * @tparam RHS A scalar or an object modeling the concept NDArray
     * @param rhs Right hand side of the = operation
     */
    template <ArrayOfRank<Rank> RHS>
    basic_array_view &operator=(RHS const &rhs) noexcept {
      // in C20 I use the concept refinement here, in 17 I have to exclude the  alternaticve
      static_assert(!is_const, "Cannot assign to a const !");
      assign_from_ndarray(rhs); // common code with view, private
      return *this;
    }

    /// Assign scalar
    template <typename RHS>
    // FIXME : explode this notion
    basic_array_view &operator=(RHS const &rhs) noexcept
      requires(is_scalar_for_v<RHS, basic_array_view>)
    {
      static_assert(!is_const, "Cannot assign to a const !");
      assign_from_scalar(rhs); // common code with view, private
      return *this;
    }

    /** 
     * 
     */
    template <ArrayInitializer<basic_array_view> Initializer>
    basic_array_view &operator=(Initializer const &initializer) noexcept {
      EXPECTS(shape() == initializer.shape());
      initializer.invoke(*this);
      return *this;
    }

    // ------------------------------- rebind --------------------------------------------

    ///
    template <typename T, int R, typename L, char A, typename AP, typename OP>
    void rebind(basic_array_view<T, R, L, A, AP, OP> v) noexcept {
      static_assert(R == Rank, "One can only rebind a view to a view of same rank");
      static_assert(std::is_same_v<std::remove_const_t<T>, std::remove_const_t<ValueType>>, "Type must be the same, except maybe const");
      static constexpr bool same_type = std::is_same_v<T, ValueType>;

      static_assert(same_type or is_const, "One can not rebind a view of T onto a view of const T. It would discard the const qualifier");
      if constexpr (same_type) {
        // FIXME Error message in layout error !
        lay = v.lay;
        sto = v.sto;
      } else if constexpr (is_const) {
        // the last if is always trivially true BUT in case of an error in the static_assert above,
        // it improves the error message by not compiling the = afterwards
        lay = layout_t{v.indexmap()};
        sto = storage_t{v.storage()};
      }
    }

    // ------------------------------- swap --------------------------------------------

    /**
     * Swaps by rebinding a and b
     * @param a
     * @param b
     */
    friend void swap(basic_array_view &a, basic_array_view &b) noexcept {
      std::swap(a.lay, b.lay);
      std::swap(a.sto, b.sto);
    }

    /**
     * Swaps the data in a and b
     * @param a
     * @param b
     */
    friend void deep_swap(basic_array_view a, basic_array_view b) noexcept {
      auto tmp = make_regular(a);
      a        = b;
      b        = tmp;
    }

#include "./_impl_basic_array_view_common.hpp"
  };

  // --- Class Template Argument Deduction Guides ---

  template <MemoryArray A>
  basic_array_view(A &&a)
     -> basic_array_view<std::conditional_t<std::is_const_v<std::remove_reference_t<A>>, const typename std::decay_t<A>::value_type,
                                            typename std::decay_t<A>::value_type>,
                         get_rank<A>, typename std::decay_t<A>::layout_policy_t, get_algebra<A>, default_accessor, borrowed<mem::get_addr_space<A>>>;

  template <typename V, size_t R>
  basic_array_view(std::array<V, R> &a)
     -> basic_array_view<V, 1, nda::basic_layout<nda::static_extents(R), nda::C_stride_order<1>, nda::layout_prop_e::contiguous>, 'V',
                         default_accessor, borrowed<>>;

  template <typename V, size_t R>
  basic_array_view(std::array<V, R> const &a)
     -> basic_array_view<const V, 1, nda::basic_layout<nda::static_extents(R), nda::C_stride_order<1>, nda::layout_prop_e::contiguous>, 'V',
                         default_accessor, borrowed<>>;

  template <std::ranges::contiguous_range R>
  basic_array_view(R &r) -> basic_array_view<std::conditional_t<std::is_const_v<R>, const typename R::value_type, typename R::value_type>, 1,
                                             C_layout, 'V', default_accessor, borrowed<>>;

  // --- get_view_t ---

  template <typename T, typename T2 = std::remove_reference_t<T> /* Keep this: Fix for gcc11 bug */>
  using get_view_t = std::remove_reference_t<decltype(basic_array_view{std::declval<T>()})>;

} // namespace nda
