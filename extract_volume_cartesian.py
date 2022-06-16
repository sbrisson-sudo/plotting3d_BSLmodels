#!/usr/bin/env python

import pickle
import numpy as np
import scipy.sparse as sparse
import json

import os

import sys
sys.path.append('/home/sbrisson/documents/Geosciences/stage-BSL/tools/ucbpy')

from Splines import CubicBSplines, SphericalSplines
from Model1D import Model1D
from ModelA3d import ModelA3d

#########
# const #
#########

# earth radius
RN = 6371.0

##########
# config #
##########

# (lower left, upper right)

# North Pacific (Hawaï large)
# box_name = '180W30S-120W30N_2800-50km'
# box = (
#     (-180.0, -30.0, 2800.0), 
#     (-120.0, 30.0, 50.0)
#     )
dx = 0.5    # degree
dr = 25.0   # km

box_name = 'hawaii_small'
box = ((-165.0, 10.0, 2800.0), (-145.0, 30.0, 50.0))
dx = 0.2
dr = 25.0

# box_name = 'indian_ocean'
# box = (
#     (40.0, -60.0, 2800.0), 
#     (120.0, 0.0, 50.0)
#     )

# box_name = 'island'
# box = (
#     (-70.0, 30.0, 2891), 
#     (60.0, 70.0, 50.0)
#     )

# model
param_grids = {'S': 'grid/grid.6', 'X': 'grid/grid.4'}
a3d_file = 'models/Model-2.6S6X4-ZeroMean.A3d'
ref_file = 'models/Model-2.6S6X4-ZeroMean_1D'

###############################################################################
###############################################################################

# sampling grid
x, y = np.meshgrid(np.arange(box[0][0], box[1][0] + 1e-5 * dx, dx),
                   np.arange(box[0][1], box[1][1] + 1e-5 * dx, dx))
x = x.T
y = y.T

r = RN - np.arange(box[0][2], box[1][2] - 1e-5 * dr, - dr)
nlon, nlat = x.shape
nr = r.size
print('volume is %i x %i x %i' % (nlon, nlat, nr))

# load the 1D reference model
ref = Model1D(ref_file)
ref.load_from_file()

# load the 3d model
a3d = ModelA3d(a3d_file)
a3d.load_from_file()

# retrieve model coefs for param of interest
p = {k: a3d.get_parameter_by_name(k) for k in param_grids}
c = {k: p[k].get_values() for k in p}

# load param knots and init spherical spline interpolation
sspl = {}
for param in param_grids:
    f = open(param_grids[param])
    f.readline()
    sspl[param] = SphericalSplines(np.loadtxt(f))
    f.close()

# init radial sampling
bspl = CubicBSplines(a3d.get_bspl_knots())
V = sparse.vstack([bspl.evaluate(k, r) for k in range(bspl.N)]).T

# init lateral sampling
sgrid = np.vstack([x.flatten(), y.flatten()]).T
H = {k: sspl[k].evaluate(sgrid) for k in sspl}

# calc reference model values
vsv = 0.001 * ref.get_values(1000 * r, parameter='vsv')
vsh = 0.001 * ref.get_values(1000 * r, parameter='vsh')
m0 = {'S': np.sqrt((2.0 * vsv ** 2 + vsh ** 2) / 3.0),
      'X': vsh ** 2 / vsv ** 2}

# sample
for k in c:
    print(H[k].shape, V.shape, c[k].shape, (V * c[k]).T.shape)
m = {k: (H[k] * (V * c[k]).T).reshape((nlon, nlat, nr)) for k in c}

# save
data_dir = "data."+box_name
print(f"Saving data in the directory {data_dir}")
if not(os.path.exists(box_name)): os.mkdir(data_dir)
os.chdir(data_dir)
pickle.dump((x, y, r, m0, m), open(f'volume.{box_name}.pkl', 'wb'))

# also save box configuration
config = {
    "box_name" : box_name,
    "box" : box,
    "dx" : dx,
    "dr" : dr
}

with open(f'box_config.{box_name}.json', 'w') as f:
    json.dump(config, f, indent=4)

