/*******************************************************************************
 *
 * TRIQS: a Toolbox for Research in Interacting Quantum Systems
 *
 * Copyright (C) 2011-2013 by O. Parcollet
 *
 * TRIQS is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * TRIQS is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * TRIQS. If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/
#pragma once
#include "./common.hpp"
#include "../storages/shared_block.hpp"
#include "./assignment.hpp"
#include "./print.hpp"
#include "../indexmaps/cuboid/foreach.hpp"
#include "triqs/utility/exceptions.hpp"
#include "triqs/utility/typeid_name.hpp"
#include "triqs/utility/view_tools.hpp"
#include <type_traits>
#ifdef TRIQS_WITH_PYTHON_SUPPORT
#include "../python/numpy_extractor.hpp"
#include "../python/array_view_to_python.hpp"
#endif

#include "triqs/h5/base_public.hpp"

namespace nda {

  template <int Rank, class T, class = std17::void_t<>> struct get_memory_layout {
    static auto invoke(T const &) { return memory_layout_t<Rank>{}; }
  };

  template <int Rank, class T> struct get_memory_layout<Rank, T, std17::void_t<decltype(std::declval<T>().memory_layout())>> {
    static auto invoke(T const &x) { return x.memory_layout(); }
  };

  template <typename A> AUTO_DECL get_shape(A const &x) RETURN(x.domain().lengths());

  template <typename A> size_t first_dim(A const &x) { return x.domain().lengths()[0]; }
  template <typename A> size_t second_dim(A const &x) { return x.domain().lengths()[1]; }
  template <typename A> size_t third_dim(A const &x) { return x.domain().lengths()[2]; }
  template <typename A> size_t fourth_dim(A const &x) { return x.domain().lengths()[3]; }
  template <typename A> size_t fifth_dim(A const &x) { return x.domain().lengths()[4]; }
  template <typename A> size_t sixth_dim(A const &x) { return x.domain().lengths()[5]; }
  template <typename A> size_t seventh_dim(A const &x) { return x.domain().lengths()[6]; }
  template <typename A> size_t eighth_dim(A const &x) { return x.domain().lengths()[7]; }
  template <typename A> size_t ninth_dim(A const &x) { return x.domain().lengths()[8]; }

  template <bool Const, typename IndexMapIterator, typename StorageType> class iterator_adapter;

  // Auxiliary class for the auto_assign of _nda_impl, proxies.
  // When implementing triqs_clef_auto_assign (A, f), if the result of f is itself a
  // clef expression, we call again triqs_clef_auto_assign.
  // This allows chain calls, cf clef adapter/vector
  // This class is moved out of _nda_impl to be reused for proxy.
  template <typename ArrayType, typename Function> struct array_auto_assign_worker {
    ArrayType &A;
    Function const &f;
    template <typename T, typename RHS> void FORCEINLINE assign(T &x, RHS &&rhs) { x = std::forward<RHS>(rhs); }
    template <typename Expr, int... Is, typename T> FORCEINLINE void assign(T &x, clef::make_fun_impl<Expr, Is...> &&rhs) {
      triqs_clef_auto_assign(x, std::forward<clef::make_fun_impl<Expr, Is...>>(rhs));
    }
    template <typename... Args> FORCEINLINE void operator()(Args const &... args) { this->assign(A(args...), f(args...)); }
  };

  // NO EXCEPT ? if stogage is [],  
  template <typename Self, typename... Args> static FORCEINLINE decltype(auto) _call_ (Self && self, Args const &... args) noexcept {
      static constexpr int Number_of_Arguments = sizeof...(Args);
      if constexpr (Number_of_Arguments == 0) return make_const_view(*this);
      if constexpr (clef::is_any_lazy_v<Args...>) { // Is it a lazy call ?
        if constexpr (R >= 0) static_assert(Number_of_Arguments == R, "Incorrect number of parameters in call");
        return make_expr_call(std::forward<Self>(self), std::forward<Args>(args)...);
      } else {                  // not lazy
      // FIXME : Clean this else, return before ?
      	if constexpr (R >= 0) { // If Rank is given at compile time, we check the number of arguments
          static constexpr bool ellipsis_is_present = ((std::is_same_v<Args, ellipsis> ? 1 : 0) + ...);
          static_assert((Number_of_Arguments == R) or (ellipsis_is_present and (Number_of_Arguments <= R)), "Incorrect number of parameters in call");
        }
        auto idx_sliced = _idx_m.slice(args...);                     // we call the index map
        if constexpr (std::is_same_v<decltype(idx_sliced), idx_map>) // Case 1 : we got a slice
          return _nda<T, idx_sliced::_Rank, make_const_view_flavor_t(Flavor), Algebra>{std::move(idx_sliced), _storage};
        else
          return _storage[idx_sliced]; // Case 2: we got a long, hence access a element
      }
    }

  //---------------
   // Algebra_t not needed in this impl. class:
  template<typename T, int Rank, flavor_t Flavor> class _nda_impl { // TRIQS_CONCEPT_TAG_NAME(MutableCuboidArray)  

    public:
    using value_type = typename StorageType::value_type;
    static_assert(!std::is_const<value_type>::value, "no const type");
    using storage_t             = StorageType;
    using _idx_m_t            = IndexMapType;
    using traversal_order_t        = typename _get_traversal_order_t<TraversalOrder>::type;
    using traversal_order_arg      = TraversalOrder;
    static constexpr int rank      = IndexMapType::domain_type::rank;
    static constexpr bool is_const = IsConst;

    static std::string hdf5_scheme() { return "array<" + triqs::h5::get_hdf5_scheme<value_type>() + "," + std::to_string(rank) + ">"; }

    protected:
    _idx_m_t _idx_m;
    storage_t _storage;

    // ------------------------------- constructors --------------------------------------------

    _nda_impl() = default;

    _nda_impl(_idx_m_t IM, storage_t ST) : _idx_m(std::move(IM)), _storage(std::move(ST)) {}

    /// The storage is allocated from the size of IM.
    _nda_impl(const _idx_m_t &IM) : _idx_m(IM), _storage() { _storage = StorageType(_idx_m.domain().number_of_elements()); }

    // Do we want to keep this ?? FIXME TRANSFORM into a maker ... 
    template <typename InitLambda>
    explicit _nda_impl(tags::_with_lambda_init, _idx_m_t IM, InitLambda &&lambda) : _idx_m(std::move(IM)), _storage() {
      _storage = StorageType(_idx_m.domain().number_of_elements(), storages::tags::_allocate_only{}); // DO NOT construct the element of the array
      _foreach_on_indexmap(_idx_m, [&](auto const &... x) { _storage._init_raw(_idx_m(x...), lambda(x...)); });
    }

    public:
    // Shallow copy
    _nda_impl(_nda_impl const &X) = default;
    _nda_impl(_nda_impl &&X)      = default;

    // ------------------------------- ==  --------------------------------------------
    // FIXME : P ULL OUT AS TEMPLATE 
    // at your own risk with floating value, but it is useful for int, string, etc....
    // in particular for tests
    friend bool operator==(_nda_impl const &A, _nda_impl const &B) {
      if (A.shape() != B.shape()) return false;
      auto ita = A.begin();
      auto itb = B.begin();
      for (; ita != A.end(); ++ita, ++itb) {
        if (!(*ita == *itb)) return false;
      }
      return true;
    }

    friend bool operator!=(_nda_impl const &A, _nda_impl const &B) { return (!(A == B)); }

    public:
    // ------------------------------- data access --------------------------------------------

    auto const &memory_layout() const { return _idx_m.memory_layout(); }
    _idx_m_t const &indexmap() const { return _idx_m; }
    storage_t const &storage() const { return _storage; }
    storage_t &storage() { return _storage; }

    /// data_start is the starting point of the data of the object
    /// this it NOT &storage()[0], which is the start of the underlying blokc
    /// they are not equal for a view in general
    value_type const *restrict data_start() const { return &_storage[_idx_m.start_shift()]; }
    value_type *restrict data_start() { return &_storage[_idx_m.start_shift()]; }

    auto const &shape() const { return _idx_m.lengths(); }

    size_t shape(size_t i) const { return _idx_m.lengths()[i]; }

    size_t num_elements() const { return _idx_m.number_of_elements(); }

    //bool is_empty() const { return this->num_elements()==0;}
    bool is_empty() const { return this->_storage.empty(); }

    // ------------------------------- operator () --------------------------------------------

    template <typename... Args> decltype(auto) operator()(Args const &... args) const & {
       return __call_impl(*this      );
    } ///
    // const &, &, && 

    private:

    /// PULL OUT ! in array and view .... STATIC 

    
  public:

   template <typename... Args> decltype(auto) operator()(Args const &... args) const & {
     return _call_(*this, args...);
   }



    template <typename... Args> decltype(auto) operator()(Args const &... args) const & {
      static constexpr int Number_of_Arguments = sizeof...(Args);
      if constexpr (Number_of_Arguments == 0) return make_const_view(*this);
      if constexpr (clef::is_any_lazy_v<Args...>) { // Is it a lazy call ?
        if constexpr (R >= 0) static_assert(Number_of_Arguments == R, "Incorrect number of parameters in call");
        return make_expr_call(*this, std::forward<Args>(args)...);
      } else {                  // not lazy
        if constexpr (R >= 0) { // If Rank is given at compile time, we check the number of arguments
          static constexpr bool ellipsis_is_present = ((std::is_same_v<Args, ellipsis> ? 1 : 0) + ...);
          static_assert((Number_of_Arguments == R) or (ellipsis_is_present and (Number_of_Arguments <= R)), "Incorrect number of parameters in call");
        }
        auto idx_sliced = _idx_m.slice(args...);                     // we call the index map
        if constexpr (std::is_same_v<decltype(idx_sliced), idx_map>) // Case 1 : we got a slice
          return _nda<T, idx_sliced::_Rank, make_const_view_flavor_t(Flavor), Algebra>{std::move(idx_sliced), _storage};
        else
          return _storage[idx_sliced]; // Case 2: we got a long, hence access a element
      }
    }

    template <typename... Args> decltype(auto) operator()(Args const &... args) & {
      static constexpr int Number_of_Arguments = sizeof...(Args);
      if constexpr (Number_of_Arguments == 0) return make_view(*this);
      if constexpr (clef::is_any_lazy_v<Args...>) { // Is it a lazy call ?
        if constexpr (R >= 0) static_assert(Number_of_Arguments == R, "Incorrect number of parameters in call");
        return make_expr_call(*this, std::forward<Args>(args)...);
      } else {                  // not lazy
        if constexpr (R >= 0) { // If Rank is given at compile time, we check the number of arguments
          static constexpr bool ellipsis_is_present = ((std::is_same_v<Args, ellipsis> ? 1 : 0) + ...);
          static_assert((Number_of_Arguments == R) or (ellipsis_is_present and (Number_of_Arguments <= R)), "Incorrect number of parameters in call");
        }
        auto idx_sliced = _idx_m.slice(args...);                     // we call the index map
        if constexpr (std::is_same_v<decltype(idx_sliced), idx_map>) // Case 1 : we got a slice
          return _nda<T, idx_sliced::_Rank, make_view_flavor_t(Flavor), Algebra>{std::move(idx_sliced), _storage};
        else
          return _storage[idx_sliced]; // Case 2: we got a long, hence access a element
      }
    }

    // TAKE A VIEW ? REALLY ??? SAHRED ONLY
    template <typename... Args> decltype(auto) operator()(Args const &... args) && {
      static constexpr int Number_of_Arguments = sizeof...(Args);
      if constexpr (Number_of_Arguments == 0) return make_view(*this);
      if constexpr (clef::is_any_lazy_v<Args...>) { // Is it a lazy call ?
        if constexpr (R >= 0) static_assert(Number_of_Arguments == R, "Incorrect number of parameters in call");
        return make_expr_call(std::move(*this), std::forward<Args>(args)...);
      } else {                  // not lazy
        if constexpr (R >= 0) { // If Rank is given at compile time, we check the number of arguments
          static constexpr bool ellipsis_is_present = ((std::is_same_v<Args, ellipsis> ? 1 : 0) + ...);
          static_assert((Number_of_Arguments == R) or (ellipsis_is_present and (Number_of_Arguments <= R)), "Incorrect number of parameters in call");
        }
        auto idx_sliced = _idx_m.slice(args...);                     // we call the index map
        if constexpr (std::is_same_v<decltype(idx_sliced), idx_map>) // Case 1 : we got a slice
          return _nda<T, idx_sliced::_Rank, make_view_flavor_t(Flavor), Algebra>{std::move(idx_sliced), _storage};
        else
          return _storage[idx_sliced]; // Case 2: we got a long, hence access a element
      }
    }

    // ------------------------------- clef auto assign --------------------------------------------

    template <typename Fnt> friend void triqs_clef_auto_assign(_nda_impl &x, Fnt f) {
      foreach (x, array_auto_assign_worker<_nda_impl, Fnt>{x, f})
        ;
    }
    // for views only !
    template <typename Fnt> friend void triqs_clef_auto_assign(_nda_impl &&x, Fnt f) {
      static_assert(IsView, "Internal errro");
      foreach (x, array_auto_assign_worker<_nda_impl, Fnt>{x, f})
        ;
    }
    // template<typename Fnt> friend void triqs_clef_auto_assign (_nda_impl & x, Fnt f) { assign_foreach(x,f);}

     protected:
      //  BOOST Serialization
    friend class boost::serialization::access;
    template <class Archive> void serialize(Archive &ar, const unsigned int version) { ar &_storage &_idx_m; }

   };
} // namespace nda
