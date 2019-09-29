import nda_py_converter_test as C
#import nda_py_a as A
import numpy as np
a = np.array(((1.0,2), (3,4)))
print a


### FIXME : make inittest

print C.f(a,1,1)

# self.assertRaises

try:
  C.f(a, 1, 2)
except:
  print "error caught"

print C.ma(5)

    
a = C.make_A()  #C.A(7)
v = a.get()
print v
v[0] *=10
print v

vc = a.get_c()
print "vc = ", vc

try:
    vc[0] *=10
except:
  print "error caught"

