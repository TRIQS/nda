#include <c2py/c2py.hpp>

#ifndef C2PY_HXX_DECLARATION_nda_converter_GUARDS
#define C2PY_HXX_DECLARATION_nda_converter_GUARDS
template <>
constexpr bool c2py::is_wrapped<nc::array_container> = true;
template <>
inline constexpr auto c2py::tp_name<nc::array_container> = "nda_converter.ArrayContainer";
#endif