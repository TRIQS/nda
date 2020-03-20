from cpp2py import Cpp2pyInfoBase

class Cpp2pyInfo(Cpp2pyInfoBase):

    table_imports = {
    }

    table_converters = {
      'nda::basic_array' : 'nda_py/cpp2py_converters.hpp',
      'nda::basic_array_view' : 'nda_py/cpp2py_converters.hpp',
    }

def _get_cpp2py_wrapped_class_enums():
    return {'module_name' : 'UNUSED', 'includes' : "['<triqs/cpp2py_converters.hpp>']"}

__all__ = ['Cpp2pyInfo']
