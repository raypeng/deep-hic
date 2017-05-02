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

import util


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

resolution = 5000
ctcf_file = 'data/fimo-output/chr21-{0}-ctcf/psum.pkl'.format(resolution)
prefix_sum = cPickle.load(open(ctcf_file, 'rb'))
ctcf, _ = util.load_ctcf_counts(prefix_sum, indices, 1)
ctcf = ctcf.reshape(-1, 1)
print 'ctcf shape', ctcf.shape
print 'reading ctcf done'

x = np.hstack((dist, ctcf))

print '\n=====\ny = f(dist, ctcf) with LogisticRegression'
lr = LR()
lr.fit(x, y)
print 'coef', lr.coef_
print 'intercept', lr.intercept_
print 'LR_r2', lr.score(x, y)

print '\n=====\ny = f(ctcf) with LogisticRegression'
lr = LR()
lr.fit(ctcf, y)
print 'coef', lr.coef_
print 'intercept', lr.intercept_
print 'LR_r2', lr.score(ctcf, y)

print '\n=====\ny = f(ctcf) with KernelRegression'
kr = KernelReg(ctcf, y, var_type='u')
print 'KR_r2', kr.r_squared()
