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
  // symmetry accepts array index and returns new array index & operation
  template <typename F, typename A, typename idx_t = std::array<long, static_cast<std::size_t>(get_rank<A>)>>
  concept NdaSymmetry = Array<A> and //
     requires(F f, idx_t const &idx) {
       { f(idx) } -> std::same_as<std::tuple<idx_t, operation>>;
     };

  // model the init function concept
  // init function accepts array index and returns array value type
  template <typename F, typename A, typename idx_t = std::array<long, static_cast<std::size_t>(get_rank<A>)>>
  concept NdaInitFunc = Array<A> and //
     requires(F f, idx_t const &idx) {
       { f(idx) } -> std::same_as<get_value_t<A>>;
     };

  // symmetry group implementation
  template <typename F, typename A>
    requires(Array<A> && NdaSymmetry<F, A>)
  class sym_grp {

    public:
    // aliases
    static constexpr int ndims = get_rank<A>;
    using sym_idx_t            = std::pair<long, operation>;
    using sym_class_t          = std::span<sym_idx_t>;

    private:
    // members
    std::vector<F> sym_list;              // list of symmetries defining the symmetry group
    std::vector<sym_class_t> sym_classes; // list of classes
    std::vector<sym_idx_t> data;          // list of all elements to have a single contigous block of memory

    public:
    // getter methods (no setter methods, members should not be modified)
    [[nodiscard]] std::vector<F> const &get_sym_list() const { return sym_list; }
    [[nodiscard]] std::vector<sym_class_t> const &get_sym_classes() const { return sym_classes; }

    // initializer method
    // iterates over symmetry classes and propagates value from initializer function
    template <typename H>
    requires(NdaInitFunc<H, A>)
    FORCEINLINE void init(A &x, H const &init_func) const {
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

      // initialize data array
      data.reserve(x.size());

      // loop over array elements and sort them into symmetry classes
      for_each(checked.shape(), [&checked, this](auto... i) {
        if (not checked(i...)) {
          // this index is now checked and generates a new symmetry class
          operation op;
          checked(i...)    = true;
          auto idx         = std::array{i...};
          auto class_start = data.end();
          this->data.emplace_back(checked.indexmap()(i...), op);

          // apply all symmetries to current index
          auto class_size = iterate(idx, op, checked);

          // add new symmetry class to list
          this->sym_classes.emplace_back(class_start, class_size);
        }
      });
    }

    private:
    long long iterate(std::array<long, static_cast<std::size_t>(get_rank<A>)> const &idx, operation const &op, array<bool, ndims> &checked,
                      long excursion_length = 0) {

      long long segment_length = 0;

      // loop over all symmetry operations
      for (auto const &sym : sym_list) {
        // apply the symmetry
        auto [idxp, opp] = sym(idx);
        opp              = opp * op;

        // check if index is valid, reset excursion_length iff we start from valid & unchecked index
        if (is_valid(checked, idxp)) {

          // check if index has been used already
          if (not std::apply(checked, idxp)) {

            // this index is now checked
            std::apply(checked, idxp) = true;

            // add to symmetry class and keep going
            this->data.emplace_back(std::apply(checked.indexmap(), idxp), opp);
            segment_length += iterate(idxp, opp, checked) + 1;
          }

        } else if (excursion_length < 10) {
          // increment excursion_length and keep going
          segment_length += iterate(idxp, opp, checked, ++excursion_length);
        }
      }

      return segment_length;
    }
  };
} // namespace nda