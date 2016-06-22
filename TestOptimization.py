import numpy as np
from scipy.optimize import minimize

# fun = lambda x: (x[0] - 1)**2 + (x[1] - 2.5)**2


def fun(x):
    return (x[0] - 1)**2 + (x[1] - 2.5)**2
cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - 2 * x[1] + 2},
        {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 6},
        {'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2})
bnds = ((0, None), (0, None))
#res = minimize(fun, (2, 0), method='SLSQP', bounds=bnds, constraints=cons)
#print res


def f(x):
    return x[0]**2 - 2*x[0] + x[1]**2

import time
def f1(x):
    print x
    return x**2 - 2*x + 1

#bods = ((1, None), (0.5, None))
#x0 = np.array([1.1, 1.2])
#res = minimize(f, x0, bounds=bods)
#print res
#print res.x

bods = [(1.9, None)]
x0 = np.array([10])
res = minimize(f1, x0, bounds=bods)
print res.x
