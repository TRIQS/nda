.. highlight:: bash

.. _install:

Compiling nda from source
===============================


Installation steps
------------------

#. Download the source code of the latest stable version by cloning the ``TRIQS/nda`` repository from GitHub::

     $ git clone https://github.com/TRIQS/nda nda.src

#. Create and move to a new directory where you will compile the code::

     $ mkdir nda.build && cd nda.build

#. In the build directory call cmake, including any additional custom CMake options, see below::

     $ cmake -DCMAKE_INSTALL_PREFIX=path_to_install_dir ../nda.src

#. Compile the code, run the tests and install the application::

     $ make
     $ make test
     $ make install

Versions
--------

To use a particular version, go into the directory with the sources, and look at all available versions::

     $ cd nda.src && git tag

Checkout the version of the code that you want::

     $ git checkout 2.1.0

and follow steps 2 to 4 above to compile the code.

Custom CMake options
--------------------

The compilation of ``nda`` can be configured using CMake-options::

    cmake ../nda.src -DOPTION1=value1 -DOPTION2=value2 ...

+-----------------------------------------------------------------+-----------------------------------------------+
| Options                                                         | Syntax                                        |
+=================================================================+===============================================+
| Specify an installation path                                    | -DCMAKE_INSTALL_PREFIX=path_to_nda            |
+-----------------------------------------------------------------+-----------------------------------------------+
| Build in Debugging Mode                                         | -DCMAKE_BUILD_TYPE=Debug                      |
+-----------------------------------------------------------------+-----------------------------------------------+
| Disable testing (not recommended)                               | -DBuild_Tests=OFF                             |
+-----------------------------------------------------------------+-----------------------------------------------+
| Build the documentation                                         | -DBuild_Documentation=ON                      |
+-----------------------------------------------------------------+-----------------------------------------------+
