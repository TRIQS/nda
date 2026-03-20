
// C.f. https://numpy.org/doc/1.21/reference/c-api/array.html#importing-the-api
#define PY_ARRAY_UNIQUE_SYMBOL _cpp2py_ARRAY_API
#ifndef CLAIR_C2PY_WRAP_GEN
#ifdef __clang__
// #pragma clang diagnostic ignored "-W#warnings"
#endif
#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wcast-function-type"
#pragma GCC diagnostic ignored "-Wcpp"
#endif

#define C2PY_VERSION_MAJOR 0
#define C2PY_VERSION_MINOR 1

#include <c2py/c2py.hpp>

using c2py::operator""_a;

// ==================== Wrapped classes =====================

template <>
constexpr bool c2py::is_wrapped<nc::array_container> = true;

// ==================== enums =====================

// ==================== module classes =====================

template <>
inline constexpr auto c2py::tp_name<nc::array_container> = "nda_converter.ArrayContainer";
static auto init_0 = c2py::dispatcher_c_kw_t{c2py::c_constructor<nc::array_container, long, long>("rows", "cols")};
template <>
constexpr initproc c2py::tp_init<nc::array_container> = c2py::pyfkw_constructor<init_0>;
template <>
const std::string c2py::tp_ctor_doc<nc::array_container> = init_0.doc(R"DOC()DOC");
// data
static auto const fun_0 = c2py::dispatcher_f_kw_t{c2py::cmethod([](nc::array_container &self) -> decltype(auto) { return self.data(); }, "self")};

// data_const
static auto const fun_1 =
   c2py::dispatcher_f_kw_t{c2py::cmethod([](nc::array_container const &self) -> decltype(auto) { return self.data_const(); }, "self")};

// data_copy
static auto const fun_2 =
   c2py::dispatcher_f_kw_t{c2py::cmethod([](nc::array_container const &self) -> decltype(auto) { return self.data_copy(); }, "self")};

// set_data
static auto const fun_3 = c2py::dispatcher_f_kw_t{
   c2py::cmethod([](nc::array_container &self,
                    nda::basic_array_view<double, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> v)
                    -> decltype(auto) { return self.set_data(v); },
                 "self", "v")};

static const auto doc_d_0 = fun_0.doc(R"DOC()DOC");
static const auto doc_d_1 = fun_1.doc(R"DOC()DOC");
static const auto doc_d_2 = fun_2.doc(R"DOC()DOC");
static const auto doc_d_3 = fun_3.doc(R"DOC()DOC");

// ----- Method table ----
template <>
PyMethodDef c2py::tp_methods<nc::array_container>[] = {
   {"data", (PyCFunction)c2py::pyfkw<fun_0>, METH_VARARGS | METH_KEYWORDS, doc_d_0.c_str()},
   {"data_const", (PyCFunction)c2py::pyfkw<fun_1>, METH_VARARGS | METH_KEYWORDS, doc_d_1.c_str()},
   {"data_copy", (PyCFunction)c2py::pyfkw<fun_2>, METH_VARARGS | METH_KEYWORDS, doc_d_2.c_str()},
   {"set_data", (PyCFunction)c2py::pyfkw<fun_3>, METH_VARARGS | METH_KEYWORDS, doc_d_3.c_str()},
   {nullptr, nullptr, 0, nullptr} // Sentinel
};

// ----- Method table ----

template <>
constinit PyGetSetDef c2py::tp_getset<nc::array_container>[] = {

   {nullptr, nullptr, nullptr, nullptr, nullptr}};

template <>
const std::string c2py::tp_doc<nc::array_container> = R"DOC()DOC" + c2py::tp_ctor_doc<nc::array_container>;

// ==================== module functions ====================

// add_arrays
static auto const fun_4 = c2py::dispatcher_f_kw_t{c2py::cfun(
   [](const nda::basic_array<long, 2, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>> &a,
      const nda::basic_array<long, 2, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>> &b) {
     return nc::add_arrays(a, b);
   },
   "a", "b")};

// conj_array
static auto const fun_5 = c2py::dispatcher_f_kw_t{c2py::cfun(
   [](const nda::basic_array<std::complex<double>, 1, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>> &a) {
     return nc::conj_array(a);
   },
   "a")};

// double_array
static auto const fun_6 = c2py::dispatcher_f_kw_t{c2py::cfun(
   [](nda::basic_array<double, 1, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>> a) {
     return nc::double_array(a);
   },
   "a")};

// fill_3d
static auto const fun_7 = c2py::dispatcher_f_kw_t{
   c2py::cfun([](nda::basic_array_view<double, 3, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> v,
                 double val) { return nc::fill_3d(v, val); },
              "v", "val")};

// fill_matrix
static auto const fun_8 = c2py::dispatcher_f_kw_t{
   c2py::cfun([](nda::basic_array_view<double, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> m,
                 double val) { return nc::fill_matrix(m, val); },
              "m", "val")};

// make_identity
static auto const fun_9 = c2py::dispatcher_f_kw_t{c2py::cfun([](long n) { return nc::make_identity(n); }, "n")};

// negate_array
static auto const fun_10 = c2py::dispatcher_f_kw_t{c2py::cfun(
   [](const nda::basic_array<double, 1, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>> &a) {
     return nc::negate_array(a);
   },
   "a")};

// scale_array
static auto const fun_11 = c2py::dispatcher_f_kw_t{
   c2py::cfun([](const nda::basic_array<double, 1, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>> &a,
                 double s) { return nc::scale_array(a, s); },
              "a", "s")};

// sum_3d
static auto const fun_12 = c2py::dispatcher_f_kw_t{c2py::cfun(
   [](const nda::basic_array<double, 3, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>> &a) {
     return nc::sum_3d(a);
   },
   "a")};

// sum_array
static auto const fun_13 = c2py::dispatcher_f_kw_t{c2py::cfun(
   [](const nda::basic_array<double, 1, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>> &a) {
     return nc::sum_array(a);
   },
   "a")};

// sum_complex_array
static auto const fun_14 = c2py::dispatcher_f_kw_t{c2py::cfun(
   [](const nda::basic_array<std::complex<double>, 1, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>> &a) {
     return nc::sum_complex_array(a);
   },
   "a")};

// sum_const_view
static auto const fun_15 = c2py::dispatcher_f_kw_t{c2py::cfun(
   [](nda::basic_array_view<const double, 1, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> v) {
     return nc::sum_const_view(v);
   },
   "v")};

// sum_int_array
static auto const fun_16 = c2py::dispatcher_f_kw_t{c2py::cfun(
   [](const nda::basic_array<long, 1, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>> &a) {
     return nc::sum_int_array(a);
   },
   "a")};

static const auto doc_d_4  = fun_4.doc(R"DOC()DOC");
static const auto doc_d_5  = fun_5.doc(R"DOC()DOC");
static const auto doc_d_6  = fun_6.doc(R"DOC()DOC");
static const auto doc_d_7  = fun_7.doc(R"DOC()DOC");
static const auto doc_d_8  = fun_8.doc(R"DOC()DOC");
static const auto doc_d_9  = fun_9.doc(R"DOC()DOC");
static const auto doc_d_10 = fun_10.doc(R"DOC()DOC");
static const auto doc_d_11 = fun_11.doc(R"DOC()DOC");
static const auto doc_d_12 = fun_12.doc(R"DOC()DOC");
static const auto doc_d_13 = fun_13.doc(R"DOC()DOC");
static const auto doc_d_14 = fun_14.doc(R"DOC()DOC");
static const auto doc_d_15 = fun_15.doc(R"DOC()DOC");
static const auto doc_d_16 = fun_16.doc(R"DOC()DOC");
//--------------------- module function table  -----------------------------

static PyMethodDef module_methods[] = {
   {"add_arrays", (PyCFunction)c2py::pyfkw<fun_4>, METH_VARARGS | METH_KEYWORDS, doc_d_4.c_str()},
   {"conj_array", (PyCFunction)c2py::pyfkw<fun_5>, METH_VARARGS | METH_KEYWORDS, doc_d_5.c_str()},
   {"double_array", (PyCFunction)c2py::pyfkw<fun_6>, METH_VARARGS | METH_KEYWORDS, doc_d_6.c_str()},
   {"fill_3d", (PyCFunction)c2py::pyfkw<fun_7>, METH_VARARGS | METH_KEYWORDS, doc_d_7.c_str()},
   {"fill_matrix", (PyCFunction)c2py::pyfkw<fun_8>, METH_VARARGS | METH_KEYWORDS, doc_d_8.c_str()},
   {"make_identity", (PyCFunction)c2py::pyfkw<fun_9>, METH_VARARGS | METH_KEYWORDS, doc_d_9.c_str()},
   {"negate_array", (PyCFunction)c2py::pyfkw<fun_10>, METH_VARARGS | METH_KEYWORDS, doc_d_10.c_str()},
   {"scale_array", (PyCFunction)c2py::pyfkw<fun_11>, METH_VARARGS | METH_KEYWORDS, doc_d_11.c_str()},
   {"sum_3d", (PyCFunction)c2py::pyfkw<fun_12>, METH_VARARGS | METH_KEYWORDS, doc_d_12.c_str()},
   {"sum_array", (PyCFunction)c2py::pyfkw<fun_13>, METH_VARARGS | METH_KEYWORDS, doc_d_13.c_str()},
   {"sum_complex_array", (PyCFunction)c2py::pyfkw<fun_14>, METH_VARARGS | METH_KEYWORDS, doc_d_14.c_str()},
   {"sum_const_view", (PyCFunction)c2py::pyfkw<fun_15>, METH_VARARGS | METH_KEYWORDS, doc_d_15.c_str()},
   {"sum_int_array", (PyCFunction)c2py::pyfkw<fun_16>, METH_VARARGS | METH_KEYWORDS, doc_d_16.c_str()},
   {nullptr, nullptr, 0, nullptr} // Sentinel
};

//--------------------- module struct & init error definition ------------

//// module doc directly in the code or "" if not present...
/// Or mandatory ?
static struct PyModuleDef module_def = {PyModuleDef_HEAD_INIT,
                                        "nda_converter",   /* name of module */
                                        R"RAWDOC()RAWDOC", /* module documentation, may be NULL */
                                        -1, /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
                                        module_methods,
                                        NULL,
                                        NULL,
                                        NULL,
                                        NULL};

//--------------------- module init function -----------------------------

extern "C" __attribute__((visibility("default"))) PyObject *PyInit_nda_converter() {

  if (not c2py::check_python_version("nda_converter")) return NULL;

  // import numpy iff 'numpy/arrayobject.h' included
#ifdef Py_ARRAYOBJECT_H
  import_array();
#endif

  PyObject *m;

  if (PyType_Ready(&c2py::wrap_pytype<c2py::py_range>) < 0) return NULL;
  if (PyType_Ready(&c2py::wrap_pytype<nc::array_container>) < 0) return NULL;

  m = PyModule_Create(&module_def);
  if (m == NULL) return NULL;

  auto &conv_table = *c2py::conv_table_sptr.get();

  conv_table[std::type_index(typeid(c2py::py_range)).name()] = &c2py::wrap_pytype<c2py::py_range>;
  c2py::add_type_object_to_main<nc::array_container>("ArrayContainer", m, conv_table);

  return m;
}
#endif
// CLAIR_WRAP_GEN
