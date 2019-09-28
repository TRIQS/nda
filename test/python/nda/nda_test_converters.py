import converter_test as C
import numpy as np
a = np.array(((1.0,2), (3,4)))
print a

print C.f(a,1,1)

# self.assertRaises

try:
  C.f(a, 1, 2)
except:
  print "error caught"

print C.ma(5)

a = C.A(7)
v = a.get()
print v
v[0] *=10
print v

vc = a.get_c()
print "vc = ", vc
vc[0] *=10
