namespace nda {
  
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

        // ------------------------------- clef auto assign --------------------------------------------

      template <typename Fnt> friend void triqs_clef_auto_assign(indexmap_storage_pair &x, Fnt f) {
        foreach (x, array_auto_assign_worker<indexmap_storage_pair, Fnt>{x, f})
          ;
      }
      // for views only !
      template <typename Fnt> friend void triqs_clef_auto_assign(indexmap_storage_pair &&x, Fnt f) {
        static_assert(IsView, "Internal errro");
        foreach (x, array_auto_assign_worker<indexmap_storage_pair, Fnt>{x, f})
          ;
      }
      // template<typename Fnt> friend void triqs_clef_auto_assign (indexmap_storage_pair & x, Fnt f) { assign_foreach(x,f);}


 // ------------------------------- ==  --------------------------------------------
  // FIXME : h5
    // static std::string hdf5_scheme() { return "array<" + triqs::h5::get_hdf5_scheme<value_type>() + "," + std::to_string(rank) + ">"; }


 // ------------------------------- ==  --------------------------------------------
    

       template <typename... INT> friend view_type transposed_view(array_view const &a, INT... is) {
      return transposed_view(a, mini_vector<int, Rank>{is...});
    }

    // OUT AS TEMPLATE MATCH
    friend view_type transposed_view(array_view const &a, mini_vector<int, Rank> const &perm) { return {transpose(a.indexmap_, perm), a.storage_}; }

    friend view_type c_ordered_transposed_view(array_view const &a) {
      return transposed_view(a, a.indexmap().memory_layout().get_memory_positions());
    }


    // eX ARRAUY
    template <typename... INT> friend const_view_type transposed_view(array const &a, INT... is) { return transposed_view(a(), is...); };
    template <typename... INT> friend view_type transposed_view(array &a, INT... is) { return transposed_view(a(), is...); };



}
// The std::swap is WRONG for a view because of the copy/move semantics of view.
// Use swap instead (the correct one, found by ADL).
namespace std {
  template <typename V, typename To1, typename To2, int R, bool B1, bool B2, bool C1, bool C2>
  void swap(triqs::arrays::array_view<V, R, To1, B1, C1> &a, triqs::arrays::array_view<V, R, To2, B2, C2> &b) = delete;
}


   // pretty print of the array
    friend std::ostream &operator<<(std::ostream &out, const _nda_impl &A) {
      if (A.storage().size() == 0)
        out << "empty ";
      else
        pretty_print(out, A);
      return out;
    }

