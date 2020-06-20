from nda_py_a import arrn, Container, make_container_array

import numpy as np

assert np.all(arrn(10) == Container(10).a)
assert np.all(arrn(10) == Container(10).get())

# Array of wrapped type
cont_arr = make_container_array()
assert cont_arr.shape == (2, 2)
assert cont_arr.dtype == np.dtype(object)
print(cont_arr[0,0].a)
print(cont_arr[0,1].a)
print(cont_arr[1,0].a)
print(cont_arr[1,1].a)
assert all(np.all(c.a == arrn(n)) for n, c in enumerate(cont_arr.flatten()))
