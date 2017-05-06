# this file performs a baseline regression of y (interaction value)
# on distance between training pairs and outputs a R2 score
#
# input to this script: h5 encoded data file


import h5py
import sys
import cPickle
import numpy as np
from statsmodels.nonparametric.kernel_regression import KernelReg
from sklearn.linear_model import LinearRegression as LR
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV

def read_val(path):
    with h5py.File(path, 'r') as f:
        dist = np.array(f.get('dist'))
        y = np.log(np.array(f.get('val')))
        indices = list(f.get('indices'))
    return dist.ravel(), y.ravel(), indices


assert len(sys.argv) == 2, 'path_to_h5'

dist, y, indices = read_val(sys.argv[1].strip())
# only regress on test set to facilitate comparison with our mode
train_frac = 0.9
n = int(y.size * (1 - train_frac)) # last 10% is used as test data
dist, y, indices = dist[-n:].reshape(-1, 1), y[-n:].reshape(-1, 1), indices[-n:]
print 'dist.shape, y.shape', dist.shape, y.shape
print 'reading h5 done'

# dist = np.random.rand(*y.shape)

print '\n=====\ny = f(dist) with LinearRegression'
lr = LR()
lr.fit(dist, y)
print 'coef', lr.coef_
print 'intercept', lr.intercept_
print 'LR_r2', lr.score(dist, y)

# s = y.size / 1000
s = 10

print '\n=====\ny = f(dist) with RBF Ridge'
kr = GridSearchCV(KernelRidge(kernel='rbf'), cv=5,
                  param_grid={'alpha': [1, 0.1, 0.01, 0.001],
                              'gamma': np.logspace(-2, 2, 5)})
kr.fit(dist[::s], y[::s])
print 'RBF_r2', kr.score(dist, y)

print '\n=====\ny = f(dist) with KernelRegression'
kr = KernelReg(dist[::s], y[::s], var_type='u')
print 'KR_r2', kr.r_squared()
