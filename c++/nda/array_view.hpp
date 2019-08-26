#pragma once
#include "./indexmap/idx_map.hpp"
#include "./storage/handle.hpp"
#include "./concepts.hpp"
#include "./assignment.hpp"

namespace nda {

  // Memory Policy
  enum class mem_policy_e { Borrowed, Shared };

  // Const or Mutable
  enum class cm_e { Const, Mutable };

  // forward
  template <typename ValueType, int Rank> class array;

  // detects ellipsis in a argument pack
  template <typename... T> constexpr bool ellipsis_is_present = ((std::is_same_v<T, ellipsis> ? 1 : 0) + ... + 0); // +0 because it can be empty

  // ---------------------- array_view  --------------------------------

  // Try to put the const/mutable in the TYPE

  //template <typename ValueType, int Rank, cm_e ConstMutable = Mutable, mem_policy_e MemPolicy = Borrowed>
  template <typename ValueType, int Rank, mem_policy_e MemPolicy = mem_policy_e::Borrowed>
  class array_view : tag::array_view // any other useful tags
  {
    //static_assert(!std::is_const<ValueType>::value, "no const type");

    public:
    using value_t                  = std::remove_const_t<ValueType>;
    static constexpr bool is_const = std::is_const_v<ValueType>;
    using value_as_template_arg_t  = ValueType;

    static constexpr mem_policy_e mem_policy = MemPolicy;
    using storage_t                          = mem::handle<value_t, (mem_policy == mem_policy_e::Shared ? 'S' : 'B')>;
    using idx_map_t                          = idx_map<Rank>;

    using regular_t    = array<value_t, Rank>;
    using view_t       = array_view<value_t, Rank>;
    using const_view_t = array_view<value_t const, Rank>;

    static constexpr int rank = Rank;

    // FIXME : h5
    // static std::string hdf5_scheme() { return "array<" + triqs::h5::get_hdf5_scheme<value_t>() + "," + std::to_string(rank) + ">"; }

    private:
    idx_map_t _idx_m;
    storage_t _storage;

    public:
    // ------------------------------- constructors --------------------------------------------

    /// Construct an empty view.
    array_view() = default;

    ///
    array_view(array_view &&) = default;

    /// Shallow copy. It copies the *view*, not the data.
    array_view(array_view const &) = default;

    /** 
     * Construct a view of T const from a view of T
     * @param v a view 
     *
     * NB : Only valid when ValueType is const
     */
    array_view(array_view<value_t, Rank> const &v) REQUIRES(is_const) : array_view(v.indexmap(), v.storage()) {}

    /**
     *  [Advanced] From an indexmap and a storage handle
     *  @param idx index map
     *  @st  storage (memory handle)
     */
    array_view(idx_map<Rank> idx, storage_t st) : _idx_m(std::move(idx)), _storage(std::move(st)) {}

    /** 
     * From anything that has an indexmap and a storage compatible with this class
     * @tparam T an array/array_view or matrix/vector type
     * @param a array or view
     *
     * NB : short cut for array_view (x.indexmap(), x.storage())
     * Allows cross construction of array_view from matrix/matrix view e.g.
     */
    template <typename T> explicit array_view(T const &a) : array_view(a.indexmap(), a.storage()) {}

    // Move assignment not defined : will use the copy = since view must copy data

    // ------------------------------- assign --------------------------------------------

    /**
     * @tparam RHS Can be 
     * 	              - an object modeling the concept NDArray
     * 	              - a type from which ValueType is constructible
     * @param rhs
     *
     * Copies the content of rhs into the view.
     * Pseudo code : 
     *     for all i,j,k,l,... : this[i,j,k,l] = rhs(i,j,k,l)
     *
     * The dimension of RHS must be large enough or behaviour is undefined.
     * 
     * If NDA_BOUNDCHECK is defined, the bounds are checked.
     */
    template <typename RHS> array_view &operator=(RHS const &rhs) {
      //nda::assignment(*this, X);
      return *this;
    }

    /// A special case of the general operator
    /// [C++ oddity : this case must be explicitly coded too]
    array_view &operator=(array_view const &rhs) {
      //nda::assignment(*this, X);
      return *this;
    }

    // ------------------------------- rebind --------------------------------------------

    /// Rebind the view
    void rebind(array_view<value_t, Rank> const &X) {
      _idx_m   = X._idx_m;
      _storage = X._storage;
    }

    /// Rebind
    void rebind(array_view<value_t const, Rank> const &X) REQUIRES(!is_const) { // REQUIRES otherwise it is the same function
      _idx_m   = X._idx_m;
      _storage = X._storage;
    }
    //check https://godbolt.org/z/G_QRCU

    // -------------------------------  operator () --------------------------------------------

    // one can factorize the last part in a private static method, but I find clearer to have the repetition
    // here. In particular to check the && case carefully.

    /// DOC
    template <typename... T> decltype(auto) operator()(T const &... x) const & {

      if constexpr (sizeof...(T) == 0)
        return view_t{*this};
      else {

        static_assert((Rank == -1) or (sizeof...(T) == Rank) or (ellipsis_is_present<T...> and (sizeof...(T) <= Rank)),
                      "Incorrect number of parameters in call");
        //if constexpr (clef::is_any_lazy_v<T...>) return clef::make_expr_call(*this, std::forward<T>(x)...);

        auto idx_or_pos = _idx_m(x...);                           // we call the index map
        if constexpr (std::is_same_v<decltype(idx_or_pos), long>) // Case 1: we got a long, hence access a element
          return _storage[idx_or_pos];                            //
        else                                                      // Case 2: we got a slice
          return view_t{std::move(idx_or_pos), _storage};         //
      }
    }
    ///
    template <typename... T> decltype(auto) operator()(T const &... x) & {

      if constexpr (sizeof...(T) == 0)
        return view_t{*this};
      else {

        static_assert((Rank == -1) or (sizeof...(T) == Rank) or (ellipsis_is_present<T...> and (sizeof...(T) <= Rank)),
                      "Incorrect number of parameters in call");
        //if constexpr (clef::is_any_lazy_v<T...>) return clef::make_expr_call(*this, std::forward<T>(x)...);

        auto idx_or_pos = _idx_m(x...);                           // we call the index map
        if constexpr (std::is_same_v<decltype(idx_or_pos), long>) // Case 1: we got a long, hence access a element
          return _storage[idx_or_pos];                            //
        else                                                      // Case 2: we got a slice
          return view_t{std::move(idx_or_pos), _storage};         //
      }
    }

    ///
    template <typename... T> decltype(auto) operator()(T const &... x) && {

      if constexpr (sizeof...(T) == 0)
        return view_t{*this};
      else {

        static_assert((Rank == -1) or (sizeof...(T) == Rank) or (ellipsis_is_present<T...> and (sizeof...(T) <= Rank)),
                      "Incorrect number of parameters in call");
        //if constexpr (clef::is_any_lazy_v<T...>) return clef::make_expr_call(std::move(*this), std::forward<T>(x)...);

        auto idx_or_pos = _idx_m(x...);                           // we call the index map
        if constexpr (std::is_same_v<decltype(idx_or_pos), long>) // Case 1: we got a long, hence access a element
          return _storage[idx_or_pos];                            // We return a REFERENCE here. Ok since underlying array is still alive
        else                                                      // Case 2: we got a slice
          return view_t{std::move(idx_or_pos), _storage};         //
      }
    }

    // ------------------------------- data access --------------------------------------------

    // The Index Map object
    idx_map<Rank> const &indexmap() const { return _idx_m; }

    // The storage handle
    storage_t const &storage() const { return _storage; }
    storage_t &storage() { return _storage; }

    // Memory layout
    auto layout() const { return _idx_m.layout(); }

    /// Starting point of the data. NB : this is NOT the beginning of the memory block for a view in general
    ValueType const *data_start() const { return _storage.data + _idx_m.offset(); }

    /// Starting point of the data. NB : this is NOT the beginning of the memory block for a view in general
    ValueType *data_start() { return _storage.data + _idx_m.offset(); }

    /// Shape of the array
    shape_t<Rank> const &shape() const { return _idx_m.lengths(); }

    /// Number of elements in the array
    long size() const { return _idx_m.size(); }

    /// FIXME : REMOVE size ? TRIVIAL
    [[deprecated]] bool is_empty() const { return size() == 0; }

    /// FIXME same as shape()[i] : redondant
    [[deprecated]] long shape(size_t i) const { return _idx_m.lengths()[i]; }

    // ------------------------------- Iterators --------------------------------------------

    //using const_iterator = iterator_adapter<true, idx_map<Rank>::iterator, storage_t>;
    //using iterator       = iterator_adapter<false, idx_map<Rank>::iterator, storage_t>;
    //const_iterator begin() const { return const_iterator(indexmap(), storage(), false); }
    //const_iterator end() const { return const_iterator(indexmap(), storage(), true); }
    //const_iterator cbegin() const { return const_iterator(indexmap(), storage(), false); }
    //const_iterator cend() const { return const_iterator(indexmap(), storage(), true); }
    //iterator begin() { return iterator(indexmap(), storage(), false); }
    //iterator end() { return iterator(indexmap(), storage(), true); }

    // ------------------------------- Operations --------------------------------------------

    //    TRIQS_DEFINE_COMPOUND_OPERATORS(array_view);

    // to forbid serialization of views...
    //template<class Archive> void serialize(Archive & ar, const unsigned int version) = delete;
  };

  /// Aliases
  template <typename ValueType, int Rank, mem_policy_e MemPolicy> using array_const_view = array_view<ValueType const, Rank, MemPolicy>;

} // namespace nda

