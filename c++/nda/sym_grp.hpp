#pragma once
#include "nda.hpp"
#include "mpi.hpp"
#include <itertools/omp_chunk.hpp>

namespace nda {

  struct operation {
    bool sgn = false; // change sign?
    bool cc  = false; // complex conjugation?

    operation operator*(operation const &x) { return operation{bool(sgn xor x.sgn), bool(cc xor x.cc)}; }

    template <typename T>
    T operator()(T const &x) const {
      if (sgn) return cc ? -conj(x) : -x;
      return cc ? conj(x) : x;
    }
  };

  template <Array A>
  bool is_valid(A const &x, std::array<long, static_cast<std::size_t>(get_rank<A>)> const &idx) {

    for (auto i = 0; i < get_rank<A>; ++i) {
      if (not(0 <= idx[i] && idx[i] < x.shape()[i])) { return false; }
    }

    return true;
  }

  // symmetry concept: symmetry accepts array index and returns new array index & operation
  template <typename F, typename A, typename idx_t = std::array<long, static_cast<std::size_t>(get_rank<A>)>>
  concept NdaSymmetry = Array<A> and //
     requires(F f, idx_t const &idx) {
       { f(idx) } -> std::same_as<std::tuple<idx_t, operation>>;
     };

  // init function concept: init function accepts array index and returns array value type
  template <typename F, typename A, typename idx_t = std::array<long, static_cast<std::size_t>(get_rank<A>)>>
  concept NdaInitFunc = Array<A> and //
     requires(F f, idx_t const &idx) {
       { f(idx) } -> std::same_as<get_value_t<A>>;
     };

  template <typename F, typename A>
    requires(Array<A> && NdaSymmetry<F, A>)
  class sym_grp {

    public:
    static constexpr int ndims = get_rank<A>;
    using sym_idx_t            = std::pair<long, operation>;
    using sym_class_t          = std::span<sym_idx_t>;
    using arr_idx_t            = std::array<long, static_cast<std::size_t>(ndims)>;

    private:
    std::vector<sym_class_t> sym_classes; // list of classes
    std::vector<sym_idx_t> data;          // long list of all elements (to allocate contigous block of memory)

    public:
    [[nodiscard]] std::vector<sym_class_t> const &get_sym_classes() const { return sym_classes; }
    long num_classes() const { return sym_classes.size(); }

    // initializer method: iterates over symmetry classes and propagates value from evaluation of init function
    template <typename H>
      requires(NdaInitFunc<H, A>)
    void init(A &x, H const &init_func, bool parallel = false) const {

      if (parallel) {
        // reset input array to allow for mpi reduction
        x() = 0.0;

        #pragma omp parallel
        for (auto const &sym_class : itertools::omp_chunk(mpi::chunk(sym_classes))) {
          auto idx           = x.indexmap().to_idx(sym_class[0].first);
          auto ref_val       = init_func(idx);
          std::apply(x, idx) = ref_val;
          for (auto const &[lin_idx, op] : sym_class) { std::apply(x, x.indexmap().to_idx(lin_idx)) = op(ref_val); }
        }

        // distribute data among all ranks
        x = mpi::all_reduce(x);

      } else {
        for (auto const &sym_class : sym_classes) {
          auto idx           = x.indexmap().to_idx(sym_class[0].first);
          auto ref_val       = init_func(idx);
          std::apply(x, idx) = ref_val;
          for (auto const &[lin_idx, op] : sym_class) { std::apply(x, x.indexmap().to_idx(lin_idx)) = op(ref_val); }
        }
      }
    }

    // symmetrization method, returns maximum symmetry violation and corresponding array index
    // NOTE: this actually requires the definition of an inverse operation, but with the current implementation
    //       operations are anyways self-inverse
    std::pair<double, arr_idx_t> symmetrize(A &x) const {

      double max_diff = 0.0;
      auto max_idx    = arr_idx_t{};

      for (auto const &sym_class : sym_classes) {
        get_value_t<A> ref_val = 0.0;

        for (auto const &[lin_idx, op] : sym_class) { ref_val += op(std::apply(x, x.indexmap().to_idx(lin_idx))); }

        ref_val /= sym_class.size();

        for (auto const &[lin_idx, op] : sym_class) {
          auto mapped_val  = op(ref_val);
          auto mapped_idx  = x.indexmap().to_idx(lin_idx);
          auto current_val = std::apply(x, mapped_idx);
          auto diff        = std::abs(mapped_val - current_val);

          if (diff > max_diff) {
            max_diff = diff;
            max_idx  = mapped_idx;
          };

          std::apply(x, mapped_idx) = mapped_val;
        }
      }

      return std::pair{max_diff, max_idx};
    }

    // constructors
    sym_grp() = default;
    sym_grp(A const &x, std::vector<F> const &sym_list, long const max_length = 0) {

      // array to check whether index has been sorted into a symmetry class already
      array<bool, ndims> checked(x.shape());
      checked() = false;

      // initialize data array (we have as many elements as in the original nda array)
      data.reserve(x.size());

      for_each(checked.shape(), [&checked, &sym_list, max_length, this](auto... i) {
        if (not checked(i...)) {
          operation op;
          checked(i...)    = true; // this index is now checked and defines a new symmetry class
          auto idx         = std::array{i...};
          auto class_start = data.end();                   // the class is added to the end of the data list
          data.emplace_back(checked.indexmap()(i...), op); // every class is initialized by one representative with op = identity

          // apply all symmetries to current index and generate the symmetry class
          auto class_size = iterate(idx, op, checked, sym_list, max_length) + 1;

          sym_classes.emplace_back(class_start, class_size);
        }
      });
    }

    private:
    // implementation of the actual symmetry reduction algorithm
    long long iterate(std::array<long, static_cast<std::size_t>(get_rank<A>)> const &idx, operation const &op, array<bool, ndims> &checked,
                      std::vector<F> const &sym_list, long const max_length, long excursion_length = 0) {

      // initialize the local segment_length to 0 (we have not advanced to a new member of the symmetry class so far)
      long long segment_length = 0;

      for (auto const &sym : sym_list) {
        // apply the symmetry
        auto [idxp, opp] = sym(idx);
        opp              = opp * op;

        // check if index is valid, reset excursion_length iff we start from valid & unchecked index
        if (is_valid(checked, idxp)) {

          // check if index has been used already
          if (not std::apply(checked, idxp)) {
            std::apply(checked, idxp) = true; // this index is now checked

            // add new member to symmetry class and increment the segment_length
            data.emplace_back(std::apply(checked.indexmap(), idxp), opp);
            segment_length += iterate(idxp, opp, checked, sym_list, max_length) + 1;
          }

          // if index is invalid, increment excursion length and keep going (segment_length is not incremented)
        } else if (excursion_length < max_length) {
          segment_length += iterate(idxp, opp, checked, sym_list, max_length, ++excursion_length);
        }
      }

      // return the final value of the local segment_length, which will be added
      // to the segment_length higher up in the recursive call tree
      return segment_length;
    }
  };
} // namespace nda