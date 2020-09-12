#pragma once
#include <vector>
#include <nda/nda.hpp>

#include "./numpy_proxy.hpp"
#include "./make_py_capsule.hpp"

// A function to make the numpy_proxy from an nda array or view
//
namespace nda::python {

  // Given an array or a view, it returns the numpy_proxy viewing its data
  // NB : accepts ref, rvalue ref
  // AUR is array<T,R> or array_view<T, R>, but NOT a the Array concept.
  // It must be a container or a view.
  template <typename AUR>
  nda::python::numpy_proxy make_numpy_proxy_from_array_or_view(AUR &&a) REQUIRES(is_regular_or_view_v<std::decay_t<AUR>>) {

    using A          = std::decay_t<AUR>;
    using value_type = typename A::value_type;          // NB May be const
    using T          = std::remove_const_t<value_type>; // The canonical type without the possible const
    static_assert(not std::is_reference_v<value_type>, "Logical Error");

    // If T is a type which has a native Numpy equivalent, or it is PyObject *  or pyref.
    // we simply take a numpy of the data
    if constexpr (nda::python::has_npy_type<T>) {
      std::vector<long> extents(A::rank), strides(A::rank);

      for (int i = 0; i < A::rank; ++i) {
        extents[i] = a.indexmap().lengths()[i];
        strides[i] = a.indexmap().strides()[i] * sizeof(T);
      }

      return {A::rank,                     // dimension
              npy_type<T>,                 // the npy type code
              (void *)a.data(),      // start of the data
              std::is_const_v<value_type>, // if const, the numpy will be immutable in python
              std::move(extents),
              std::move(strides),
              make_pycapsule(a.storage())}; // a PyCapsule with a SHARED view on the data
    } else {
      // If T is another type, which requires some conversion to python
      // we make a new array of pyref and return a numpy view of it.
      // each pyref handles the ownership of the object according to the T conversion rule, 
      // it is not the job of this function to handle this.
      // We need to distinguish the special case where a is a RValue, in which case, the python will steal the ownership
      // by moving the elements one by one.
      //nda::array<cpp2py::pyref, A::rank> aobj = map([](auto &&x) {
      nda::array<pybind11::object, A::rank> aobj = map([](auto &&x) {
        if constexpr (is_view_v<A> or std::is_reference_v<AUR>)
          return pybind11::cast(x); //cpp2py::py_converter<T>::c2py(x);
        else // nda::array rvalue (i.e. AUR is an array, and NOT a ref, so it matches array &&. Be sure to move
          return pybind11::cast(std::move(x)); //  cpp2py::py_converter<T>::c2py(std::move(x));
      })(a);
      return make_numpy_proxy_from_array_or_view(std::move(aobj));
    }
  }

} // namespace nda::python

