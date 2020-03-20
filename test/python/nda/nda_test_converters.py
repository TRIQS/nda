from nda_py_a import arrn, Container

import numpy as np

assert np.all(arrn(10) == Container(10).a)
assert np.all(arrn(10) == Container(10).get())
