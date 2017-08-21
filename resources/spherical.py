#!/usr/bin/env python

import os
import sys

import numpy as np
import scipy.sparse as sps

from clebsch import CG_tabulate
from HoomdAnalysis import Analyser


##################################
## Input/output                 ##
##################################

if len(sys.argv) != 5:
	print("\033[1;31mUsage is %s trajectory n_eq l_max n_bins\033[0m" % sys.argv[0])
	
	sys.exit()

file_traj = os.path.realpath(sys.argv[1])

n_eq      = int(sys.argv[2])
l_max     = int(sys.argv[3])
n_bins    = int(sys.argv[4])

if l_max % 2 != 0:
	l_max -= 1
	
	print("Warning: l_max is odd - rounding down to %d" % l_max)

r_min     = 0.
r_max     = 12.

n_theta   = 100
l_print   = min(2, l_max)


##################################
## Load HoomdAnalyser class     ##
##################################

path_traj = os.path.dirname(file_traj)

a         = Analyser(file_traj)
sph_inds  = np.vectorize(a.sph_idx)


##################################
## Compute harmonic indices     ##
##################################

inds = []

# Prune symmetrically-redundant indices
for l1 in range(l_max+1):
	if l1 % 2 == 0:
		for m1 in range(l1+1):
			for l2 in range(l_max+1):
				if l2 % 2 == 0:
					for m2 in range(-l2, l2+1):
						for l in range(abs(m1+m2), l_max+1):
							if l % 2 == 0:
								if m1 == 0:
									if m1+m2 >= 0: inds.append([l1,m1,l2,m2,l,-m1-m2])
								
								else: inds.append([l1,m1,l2,m2,l,-m1-m2])

inds  = np.asarray(inds, dtype=np.int32)

n_tot = len(inds)
n_sh  = sph_inds(l_max, l_max)+1

# Work out degeneracies
deg   = np.zeros(n_tot)

for idx in range(n_tot):
	l1,m1,l2,m2,l,m = inds[idx]
	
	if m1 == m2 == 0: deg[idx] = 1
	else:             deg[idx] = 2

# Tabulate relevant Clebsch-Gordan coefficients
CGs = CG_tabulate(l_max)

print("\033[1;36mPrecomputed %d Clebsch-Gordan coefficients\033[0m" % np.size(CGs))


##################################
## Ensemble averages            ##
##################################

# Average single particle harmonics, order parameter and box volumes
f          = a.accumulate(a.single_sh_aves, n_eq, l_max=l_max)
dims       = a.accumulate(a.box_dims,  n_eq)

f          = np.mean(f, axis=0)

ops        = a.accumulate(a.nematic_q, n_eq)
op_ave     = np.mean(ops)

vols       = dims[:,0]*dims[:,1]*dims[:,2]
rho        = np.mean(a.n_part/vols)

# Save projected orientation distribution function
psi        = np.zeros([n_theta,2], dtype=np.float32)
theta_grid = np.linspace(0, np.pi, num=n_theta, dtype=np.float32)

psi[:,0]   = theta_grid

for idx_theta,theta in enumerate(theta_grid):
	for l in range(l_max+1):
		if l % 2 == 0: psi[idx_theta,1] += np.real(f[l]*a.get_sph_harm(l, 0, theta, 0))

file_psi = "%s/psi_%d_%d.res" % (path_traj, n_eq, l_max)
np.savetxt(file_psi, psi)

print("Order parameter: %f, <Y20>: %f" % (op_ave, np.real(f[2]*np.sqrt(4*np.pi/5.))))
print("\033[1;32mpsi printed to '%s'\033[0m" % file_psi)

# Compute pair correlation function
bins,gr     = a.g_hist(n_eq, n_bins=n_bins, r_min=r_min, r_max=r_max)

# Average pair spherical harmonics
rho2        = a.average(a.pair_sh_aves, n_eq, bins=bins, inds=inds)
rho2       *= 4*np.pi*rho**2 * gr[:,None]

# Save pair spherical harmonics up to rank l_print
path        = "%s/harmonics" % path_traj
if not os.path.exists(path): os.makedirs(path)

rho2_r      = np.zeros([n_bins,2], dtype=np.float32)
rho2_r[:,0] = bins[:-1]

ctr         = 0

for idx in range(n_tot):
	l1,m1,l2,m2,l,m = inds[idx]
	
	if ( (l1 < l_print+1) & (l2 < l_print+1) & (l < l_print+1) ):
		rho2_r[:,1] = rho2[:,idx]

		file_rho    = "%s/rho_%d_%d%d_%d%d_%d%d.res" % (path,n_eq,l1,m1,l2,m2,l,m)
		np.savetxt(file_rho, rho2_r)

		ctr += 1

print("\033[1;32m%d rho coefficients printed in '%s/'\033[0m" % (ctr,path))


##################################
## H-matrix inversion           ##
##################################

# Invert H-equation in the form rho2_r = alpha*h_r + v, with rho2_r,h_r,v of size n_tot
h     = np.zeros_like(rho2, dtype=np.float32)

# Rescale rho2 by relevant degeneracies
rho2 *= deg[None,:]

# Symmetrised Clebsch-Gordan sums
def CG_sum(l1, l2, lp1, lp2, m1, m2):
	c_sum = 0.
	
	if ( (lp1 >= abs(m1)) & (lp2 >= abs(m2)) ):
		lpps  = np.arange(0, l_max+1, 2)
	
		cgs1  = CGs[sph_inds(l1,m1),sph_inds(lp1,m1),sph_inds(lpps,0)] * f[::2]
		cgs2  = CGs[sph_inds(l2,m2),sph_inds(lp2,m2),sph_inds(lpps,0)] * f[::2]
	
		c_sum = np.sum(np.outer(cgs1,cgs2))
	
	return c_sum

# Compute v
v = np.zeros(n_tot, dtype=np.float32)

for idx in range(n_tot):
	l1,m1,l2,m2,l,m = inds[idx]
	
	if ( (m1 == 0) & (m2 == 0) & (l == 0) & (m == 0) ):
		coeff  = rho**2 * np.sqrt(4*np.pi) * f[l1]*f[l2]
		v[idx] = coeff

# Compute alpha in sparse matric format
rows,cols,data = [],[],[]

for idx1 in range(n_tot):
	l1,m1,l2,m2,l,m = inds[idx1]

	for idx2 in range(n_tot):
		lp1,mp1,lp2,mp2,lp,mp = inds[idx2]

		if ( (mp1 == m1) & (mp2 == m2) & (lp == l) & (mp == m) ):
			coeff = rho**2 * CG_sum(l1, l2, lp1, lp2, m1, m2) * deg[idx2]

			rows.append(idx1)
			cols.append(idx2)
			data.append(coeff)

alpha = sps.coo_matrix((data, (rows,cols)), shape=(n_tot,n_tot)).tocsc()

# Invert alpha and solve for h
alpha_inv = sps.linalg.inv(alpha)

for idx_r,rho2_r in enumerate(rho2): h[idx_r,:] = alpha_inv.dot(rho2_r-v)

# Print h coefficients up to rank l_print
h_res      = np.zeros([n_bins,2], dtype=np.float32)
h_res[:,0] = bins[:-1]

for idx in range(n_tot):
	l1,m1,l2,m2,l,m = inds[idx]
	
	if ( (l1 < l_print+1) & (l2 < l_print+1) & (l < l_print+1) ):
		h_res[:,1] = h[:,idx]
		
		file_h = "%s/h_%d_%d_%d%d_%d%d_%d%d.res" % (path,n_eq,l_max,l1,m1,l2,m2,l,m)
		np.savetxt(file_h, h_res)

print("\033[1;32m%d h coefficients printed in '%s/'\033[0m" % (ctr,path))
