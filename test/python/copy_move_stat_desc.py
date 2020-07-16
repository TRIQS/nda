# Generated automatically using the command :
# c++2py copy_move_stat.hpp -C nda_py -m copy_move_stat -o copy_move_stat --cxxflags="-std=c++17" --includes=./../../c++ --includes=./../../python --target_file_only
from cpp2py.wrap_generator import *

# The module
module = module_(full_name = "copy_move_stat", doc = r"", app_name = "copy_move_stat")

# Imports

# Add here all includes
module.add_include("copy_move_stat.hpp")

# Add here anything to add in the C++ code at the start, e.g. namespace using
module.add_preamble("""
#include <nda_py/cpp2py_converters.hpp>

""")


# The class copy_move_stat
c = class_(
        py_type = "CopyMoveStat",  # name of the python class
        c_type = "copy_move_stat",   # name of the C++ class
        doc = r"""""",   # doc of the C++ class
        hdf5 = False,
)

c.add_constructor("""(bool verbose = true)""", doc = r"""""")

c.add_method("""long copy_count ()""",
             is_static = True,
             doc = r"""""")

c.add_method("""long move_count ()""",
             is_static = True,
             doc = r"""""")

c.add_method("""void reset ()""",
             is_static = True,
             doc = r"""""")

module.add_class(c)

# The class member_stat
c = class_(
        py_type = "MemberStat",  # name of the python class
        c_type = "member_stat",   # name of the C++ class
        doc = r"""""",   # doc of the C++ class
        hdf5 = False,
)

c.add_member(c_name = "m",
             c_type = "copy_move_stat",
             read_only= False,
             doc = r"""""")

c.add_constructor("""()""", doc = r"""""")

module.add_class(c)

module.add_function ("copy_move_stat make_obj ()", doc = r"""""")

module.add_function ("nda::array<copy_move_stat, 1> make_arr (long n)", doc = r"""""")

module.add_function ("nda::array<copy_move_stat, 2> make_arr (long n1, long n2)", doc = r"""""")

module.add_function ("long take_obj (copy_move_stat o)", doc = r"""""")

module.add_function ("long take_arr (nda::array<copy_move_stat, 1> a)", doc = r"""""")

module.add_function ("long take_arr (nda::array<copy_move_stat, 2> a)", doc = r"""""")



module.generate_code()