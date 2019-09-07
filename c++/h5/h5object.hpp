#pragma once
#include <complex>
#include <sstream>
#include "./macros.hpp"

namespace h5 {

  // We copy this from hdf5.h, and static_assert its validity in the cpp
  // in order to completely isolate our header from the hdf5 headers
  // Hence complex installation paths to hdf5 are only needed in the cpp file,
  // not by the users of the library.
  using hid_t   = int64_t;
  using hsize_t = unsigned long long;
  using v_t = std::vector<hsize_t>;

  // Correspondance T -> hdf5 type
  template <typename T> hid_t hdf5_type;

  //
  template <typename T> struct _is_complex : std::false_type {};
  template <typename T> struct _is_complex<std::complex<T>> : std::true_type {};
  template <typename T> constexpr bool is_complex_v = _is_complex<T>::value;

  //
  template <typename... T> std::runtime_error make_runtime_error(T const &... x) {
    std::stringstream fs;
    (fs << ... << x);
    return std::runtime_error{fs.str()};
  }

  //------------- general hdf5 object ------------------
  // HDF5 uses a reference counting system, similar to python.
  // This is a handle to an HDF5 Object, with the proper ref. counting
  // using a RAII pattern.
  // We are going to store the id of the various h5 object in such a wrapper
  // to provide clean decref, and h5_exception safety.
  // DO NOT DOCUMENT : not for users
  class h5_object {

    protected:
    hid_t id = 0;

    public:
    // make an h5_object when the id is now owned (simply inc. the ref).
    static h5_object from_borrowed(hid_t id);

    /// Constructor from an owned id (or 0). It will NOT incref, it takes ownership
    h5_object(hid_t id = 0) : id(id) {}

    /// A new ref. No deep copy.
    h5_object(h5_object const &x);

    /// Steal the reference
    h5_object(h5_object &&x) noexcept : id(x.id) { x.id = 0; }

    ///
    h5_object &operator=(h5_object const &x) { return operator=(h5_object(x)); } //rewriting with the next overload

    ///
    h5_object &operator=(h5_object &&x) noexcept; //steals the ref, after properly decref its own.

    ///
    ~h5_object();

    /// Release the HDF5 handle and reset the object to default state (id =0).
    void close();

    /// cast operator to use it in the C function as its id
    operator hid_t() const { return id; }

    /// Ensure the id is valid (by H5Iis_valid).
    bool is_valid() const;
  };

  //-----------------------------

  // simple but useless aliases
  using dataset   = h5_object;
  using datatype  = h5_object;
  using dataspace = h5_object;
  using proplist  = h5_object;
  using attribute = h5_object;

}; // namespace h5
