# Generated automatically using the command :
# c++2py ../../c++/nda/toto.hpp -p --members_read_only -N nda -a nda -m toto_module -o toto_module -C pytriqs --cxxflags="-std=c++17"
from cpp2py.wrap_generator import *

# The module
module = module_(full_name = "toto_module", doc = "", app_name = "nda")

# Imports

# Add here all includes
module.add_include("nda/toto.hpp")

# Add here anything to add in the C++ code at the start, e.g. namespace using
module.add_preamble("""
#include <cpp2py/converters/string.hpp>
#include <triqs/cpp2py_converters/h5.hpp>

using namespace nda;
""")


# The class toto
c = class_(
        py_type = "Toto",  # name of the python class
        c_type = "nda::toto",   # name of the C++ class
        doc = """A very useful and important class\n\n @note A Useful note""",   # doc of the C++ class
        hdf5 = True,
        arithmetic = ("add_only"),
        comparisons = "==",  
        serializable = "tuple",
)

c.add_constructor("""()""", doc = """""")

c.add_constructor("""(int i_)""", doc = """Construct from integer\n\n :param i_: a scalar""")

c.add_method("""std::string hdf5_scheme ()""",
             is_static = True,
             doc = """HDF5""")

c.add_property(name = "i",
               getter = cfunction("int get_i ()"),
               doc = """Simple accessor""")

module.add_class(c)

module.add_function ("int nda::chain (int i, int j)", doc = """Chain digits of two integers\n\n Chain the decimal digits of two integers i and j, and return a new \n\n @param :math:`i` The first integer\n @param :math:`j` The second integer \n @return An integer containing the digits of both i and j\n\n @remark""")



module.generate_code()
