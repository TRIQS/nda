# Generated automatically using the command :
# c++2py a.hpp -C nda_py -a nda_py_a -m nda_py_a -o nda_py_a --cxxflags="-std=c++17" --includes=./../../../c++ --includes=./../../../python --target_file_only
from cpp2py.wrap_generator import *

# The module
module = module_(full_name = "nda_py_a", doc = r"", app_name = "nda_py_a")

# Imports

# Add here all includes
module.add_include("a.hpp")

# Add here anything to add in the C++ code at the start, e.g. namespace using
module.add_preamble("""
#include <nda_py/cpp2py_converters.hpp>

""")


# The class container
c = class_(
        py_type = "Container",  # name of the python class
        c_type = "container",   # name of the C++ class
        doc = r"""""",   # doc of the C++ class
        hdf5 = False,
)

c.add_member(c_name = "a",
             c_type = "nda::array<long, 1>",
             read_only= False,
             doc = r"""""")

c.add_constructor("""(int n)""", doc = r"""""")

c.add_method("""nda::array_view<long, 1> get ()""",
             doc = r"""""")

module.add_class(c)

module.add_function ("nda::array<long, 1> arrn (int n)", doc = r"""""")



module.generate_code()
