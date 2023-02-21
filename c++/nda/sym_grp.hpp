#pragma once

namespace nda {
  struct operation {
    // members
    bool sgn = false; // change sign?
    bool cc  = false; // complex conjugation?

    // define multiplication to chain operations together
    operation operator*(operation const &x) { return operation{bool(sgn xor x.sgn), bool(cc xor x.cc)}; }

    // define how operator acts on given value
    template <typename T>
    std::complex<double> operator()(T const &x) {
      if (sgn) return cc ? -conj(x) : -x;
      return cc ? conj(x) : x;
    }
  };

  template <nda::Array A>
  FORCEINLINE bool is_valid(A const &x, std::array<long, static_cast<std::size_t>(nda::get_rank<A>)> const &idxs) {

    // check that indices are valid for each dimension
    for (auto i = 0; i < nda::get_rank<A>; ++i) {
      if (not(0 <= idxs[i] && idxs[i] < x.shape()[i])) { return false; }
    }

    return true;
  }

  template <nda::Array A>
  class sym_grp {
    public:
    // aliases
    static constexpr int ndims = nda::get_rank<A>;
    using idx_t                = std::array<long, static_cast<std::size_t>(ndims)>;
    using sym_idx_t            = std::pair<idx_t, operation>;
    using sym_func_t           = std::function<operation(idx_t &)>; // symmetry mutates index and returns operation
    using sym_class_t          = std::vector<sym_idx_t>;            // symmetry class

    private:
    // members
    std::vector<sym_func_t> sym_list;     // list of symmetries defining the symmetry group
    std::vector<sym_class_t> sym_classes; // list of symmetric elements

    public:
    // getter methods (no setter methods, members should not be modified)
    [[nodiscard]] std::vector<sym_func_t> const &get_sym_list() const { return sym_list; }
    [[nodiscard]] std::vector<sym_class_t> const &get_sym_classes() const { return sym_classes; }

    // initializer method 1
    // iterates over symmetry classes and propagates first element
    void init(A &x) {
      for (auto sym_class : sym_classes) {
        auto ref_val = std::apply(x, sym_class[0].first);

        for (auto idx = 1; idx < sym_class.size(); ++idx) { std::apply(x, sym_class[idx].first) = sym_class[idx].second(ref_val); }
      }
    }

    // initializer method 2
    // iterates over symmetry classes and propagates value from initializer function
    template <typename T>
    void init(A &x, std::function<T(idx_t const &)> init_func) {
      for (auto sym_class : sym_classes) {
        auto ref_val                      = init_func(sym_class[0].first);
        std::apply(x, sym_class[0].first) = ref_val;

        for (auto idx = 1; idx < sym_class.size(); ++idx) { std::apply(x, sym_class[idx].first) = sym_class[idx].second(ref_val); }
      }
    }

    // symmetrization, similar to initializer method 1 but with error estimate
    double symmetrize(A &x) {
      double max_diff = 0.0;

      for (auto sym_class : sym_classes) {
        auto ref_val = std::apply(x, sym_class[0].first);

        for (auto idx = 1; idx < sym_class.size(); ++idx) {
          double diff = std::abs(std::apply(x, sym_class[idx].first) - sym_class[idx].second(ref_val));
          if (diff > max_diff) { max_diff = diff; };
          std::apply(x, sym_class[idx].first) = sym_class[idx].second(ref_val);
        }
      }

      return max_diff;
    }

    // constructor
    sym_grp(A const &x, std::vector<sym_func_t> const &sym_list_) : sym_list(sym_list_) {

      // array to check whether index has been sorted into a symmetry class already
      nda::array<bool, ndims> checked(x.shape());
      checked() = false;

      // loop over array elements and sort them into symmetry classes
      nda::for_each(checked.shape(), [&checked, this](auto... i) {
        if (not checked(i...)) {
          // this index is now checked and generates a new symmetry class
          checked(i...) = true;
          sym_class_t sym_class;
          auto idx = std::array{i...};
          operation op;
          sym_class.push_back({idx, op});

          // apply all symmetries to current index
          iterate(idx, op, checked, sym_class);

          // add new symmetry class to list
          this->sym_classes.push_back(sym_class);
        }
      });
    }

    private:
    void iterate(idx_t const &idx, operation const &op, nda::array<bool, ndims> &checked, sym_class_t &sym_class, long path_length = 0) {
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
            sym_class.push_back({idxp, opp});
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