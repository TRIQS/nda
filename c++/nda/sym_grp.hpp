#pragma once

namespace nda {
  // residual operation after applying a symmetry
  struct operation {
    // members
    bool sgn = false; // change sign?
    bool cc  = false; // complex conjugation?

    // define multiplication to chain operations together
    operation operator*(operation const &x) { return operation{bool(sgn xor x.sgn), bool(cc xor x.cc)}; }

    // define how operator acts on given value
    template <typename T>
    T operator()(T const &x) {
      if (sgn) return cc ? -conj(x) : -x;
      return cc ? conj(x) : x;
    }
  };

  // check if array index is valid (move into basic_array.hpp?)
  template <Array A>
  FORCEINLINE bool is_valid(A const &x, std::array<long, static_cast<std::size_t>(get_rank<A>)> const &idx) {

    // check that indices are valid for each dimension
    for (auto i = 0; i < get_rank<A>; ++i) {
      if (not(0 <= idx[i] && idx[i] < x.shape()[i])) { return false; }
    }

    return true;
  }

  // model the symmetry concept
  // symmetry mutates array index and returns operation (make idx const and return new idx explicitly?)
  template <typename F, typename A>
  concept NdaSymmetry = Array<A> and requires(F f, std::array<long, static_cast<std::size_t>(get_rank<A>)> &idx) {
    { f(idx) } -> std::same_as<operation>;
  };

  // model the init function concept
  // init function accepts array index and returns array value type
  template <typename F, typename A>
  concept NdaInitFunc = Array<A> and requires(F f, std::array<long, static_cast<std::size_t>(get_rank<A>)> const &idx) {
    { f(idx) } -> std::same_as<get_value_t<A>>;
  };

  // symmetry group implementation
  template <typename F, typename A>
  requires(Array<A> &&NdaSymmetry<F, A>) class sym_grp {
    public:
    // aliases
    static constexpr int ndims = get_rank<A>;
    using sym_idx_t            = std::pair<long, operation>;
    using sym_class_t          = std::vector<sym_idx_t>;

    private:
    // members
    std::vector<F> sym_list;              // list of symmetries defining the symmetry group
    std::vector<sym_class_t> sym_classes; // list of symmetric elements

    public:
    // getter methods (no setter methods, members should not be modified)
    [[nodiscard]] std::vector<F> const &get_sym_list() const { return sym_list; }
    [[nodiscard]] std::vector<sym_class_t> const &get_sym_classes() const { return sym_classes; }

    // initializer method 1
    // iterates over symmetry classes and propagates first element
    FORCEINLINE void init(A &x) const {
      for (auto sym_class : sym_classes) {
        auto ref_val = std::apply(x, x.indexmap().to_idx(sym_class[0].first));
        for (auto idx = 1; idx < sym_class.size(); ++idx) {
          std::apply(x, x.indexmap().to_idx(sym_class[idx].first)) = sym_class[idx].second(ref_val);
        }
      }
    }

    // initializer method 2
    // iterates over symmetry classes and propagates value from initializer function
    template <typename H>
    requires(NdaInitFunc<H, A>) FORCEINLINE void init(A &x, H const &init_func) const {
      for (auto sym_class : sym_classes) {
        auto idx           = x.indexmap().to_idx(sym_class[0].first);
        auto ref_val       = init_func(idx);
        std::apply(x, idx) = ref_val;
        for (auto i = 1; i < sym_class.size(); ++i) { std::apply(x, x.indexmap().to_idx(sym_class[i].first)) = sym_class[i].second(ref_val); }
      }
    }

    // constructor
    sym_grp(A const &x, std::vector<F> const &sym_list_) : sym_list(sym_list_) {

      // array to check whether index has been sorted into a symmetry class already
      array<bool, ndims> checked(x.shape());
      checked() = false;

      // loop over array elements and sort them into symmetry classes
      for_each(checked.shape(), [&checked, this](auto... i) {
        if (not checked(i...)) {
          // this index is now checked and generates a new symmetry class
          checked(i...) = true;
          sym_class_t sym_class;
          operation op;
          auto idx = std::array{i...};
          sym_class.push_back({checked.indexmap()(i...), op});

          // apply all symmetries to current index
          iterate(idx, op, checked, sym_class);

          // add new symmetry class to list
          this->sym_classes.push_back(sym_class);
        }
      });
    }

    private:
    void iterate(std::array<long, static_cast<std::size_t>(get_rank<A>)> const &idx, operation const &op, array<bool, ndims> &checked,
                 sym_class_t &sym_class, long path_length = 0) {

      // loop over all symmetry operations
      for (auto sym : sym_list) {
        // copy the index before mutating it
        auto idxp = idx;

        // apply the symmetry operation (mutates idxp)
        auto opp = sym(idxp) * op;

        // check if index is valid, reset path_length iff we start from valid & unchecked index
        if (is_valid(checked, idxp)) {
          // check if index has been used already
          if (not std::apply(checked, idxp)) {
            // this index is now checked
            std::apply(checked, idxp) = true;

            // add to symmetry class and keep going
            sym_class.push_back({std::apply(checked.indexmap(), idxp), opp});
            iterate(idxp, opp, checked, sym_class);
          }
        } else if (path_length < 10) {
          // increment path_length and keep going
          iterate(idxp, opp, checked, sym_class, ++path_length);
        }
      }
    }
  };
} // namespace nda