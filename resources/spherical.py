#!/usr/bin/env python

import os
import sys

import numpy as np
import HoomdAnalysis as ha

from sympy.physics.quantum.cg import CG


# Input/output
if len(sys.argv) != 5:
	print("\033[1;31mUsage is %s trajectory n_eq l_max n_bins\033[0m" % sys.argv[0])
	sys.exit()


file_traj  = os.path.realpath(sys.argv[1])

n_eq       = int(sys.argv[2])
l_max      = int(sys.argv[3])
n_bins     = int(sys.argv[4])

n_theta    = 100

r_min      = 0.
r_max      = 12.

if l_max % 2 != 0:
	l_max -= 1
	print("Warning: l_max is odd - rounding down to %d" % l_max)

l_print    = min(2, l_max)


path_traj  = os.path.dirname(file_traj)
a          = ha.Analyser(file_traj)


# Average single particle harmonics, order parameter and box volumes
f          = a.accumulate(a.single_sh_aves, n_eq, log=True, l_max=l_max)
dims       = a.accumulate(a.box_dims,  n_eq)

f          = np.mean(f, axis=0)

ops        = a.accumulate(a.nematic_q, n_eq)
op_ave     = np.mean(ops)

vols       = dims[:,0]*dims[:,1]*dims[:,2]
rho        = np.mean(a.n_part/vols)

# Save projected orientation distribution function
theta_grid = np.linspace(0, np.pi, num=n_theta)
psi        = np.zeros([n_theta,2])

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
rho2        = a.average(a.pair_sh_aves, n_eq, log=True, bins=bins, l_max=l_max)
rho2       *= 4*np.pi*rho**2 * gr[:,None,None,None]

# Save pair spherical harmonics up to rank l_print
path        = "%s/harmonics" % path_traj
if not os.path.exists(path): os.makedirs(path)

p_real      = np.zeros([n_bins,2])
p_imag      = np.zeros([n_bins,2])

p_real[:,0] = bins[:-1]
p_imag[:,0] = bins[:-1]

ctr         = 0

for l1 in range(l_print+1):
	for l2 in range(l_print+1):
		for l in range(l_print+1):
			if ( (l1 % 2 == 0) & (l2 % 2 == 0) & (l % 2 == 0) ):
				for m1 in range(-l1, l1+1):
					for m2 in range(-l2, l2+1):
						for m in range(-l, l+1):
							p_ave = rho2[:,a.sph_idx(l1,m1),a.sph_idx(l2,m2),a.sph_idx(l,m)]

							p_real[:,1] = np.real(p_ave)
							p_imag[:,1] = np.imag(p_ave)

							file_real = "%s/real_%d_%d%d_%d%d_%d%d.res" % (path,n_eq,l1,m1,l2,m2,l,m)
							file_imag = "%s/imag_%d_%d%d_%d%d_%d%d.res" % (path,n_eq,l1,m1,l2,m2,l,m)

							np.savetxt(file_real, p_real)
							np.savetxt(file_imag, p_imag)

							ctr += 2

print("\033[1;32m%d spherical averages printed in '%s/'\033[0m" % (ctr,path))


# Renormalised Clebsch-Gordon coefficients
def CG_norm(l, lp, lpp, m, mp, mpp):
	coeff1 = CG(lpp, mpp, lp, mp, l, m)
	coeff2 = CG(lpp, 0,   lp, 0,  l, 0)

	coeff1 = float(coeff1.doit())
	coeff2 = float(coeff2.doit())

	return np.sqrt((2.*lpp+1)*(2.*lp+1)/(4*np.pi*(2.*l+1))) * coeff1*coeff2

# Resummed Clebsch-Gordon coefficients
def CG_sum(l1, l2, lp1, lp2, m1, m2):
	sum = 0.

	for lpp1 in range(l_max+1):
		for lpp2 in range(l_max+1):
			if ( (lpp1 % 2 == 0) & (lpp2 % 2 == 0) ):
				cgs  = CG_norm(l1, lp1, lpp1, m1, m1, 0) * CG_norm(l2, lp2, lpp2, m2, m2, 0)
				sum += f[lpp1]*f[lpp2] * cgs

	return sum


# Cast total correlation coefficients in the form rho2(r) = alpha*h(r)+v
rho2  = np.real(rho2)
f     = np.real(f)

h     = np.zeros_like(rho2)

# Compute v
v     = np.zeros_like(rho2[0,...])

for l1 in range(l_max+1):
	for l2 in range(l_max+1):
		if ( (l1 % 2 == 0) & (l2 % 2 == 0) ):
			coeff = rho**2 * np.sqrt(4*np.pi) * f[l1]*f[l2]
			v[a.sph_idx(l1,0),a.sph_idx(l2,0),0] = coeff


# Compute alpha
v     = v.flatten()

n_tot = len(v)
n_sh  = a.sph_idx(l_max, l_max)+1

alpha = np.zeros([n_tot,n_tot])

for l1 in range(l_max+1):
	for l2 in range(l_max+1):
		for l in range(l_max+1):
			if ( (l1 % 2 == 0) & (l2 % 2 == 0) & (l % 2 == 0) ):
				for m1 in range(-l1, l1+1):
					for m2 in range(-l2, l2+1):
						for m in range(-l, l+1):
							row = np.zeros_like(rho2[0,...])
							idx = a.sph_idx(l1,m1)*n_sh**2 + a.sph_idx(l2,m2)*n_sh + a.sph_idx(l,m)
			
							for lp1 in range(abs(m1), l_max+1):
								for lp2 in range(abs(m2), l_max+1):
									if ( (lp1 % 2 == 0) & (lp2 % 2 == 0) ):
										row[a.sph_idx(lp1,m1),a.sph_idx(lp2,m2),a.sph_idx(l,m)] = CG_sum(l1,l2,lp1,lp2,m1,m2)

							alpha[idx,:] = rho**2 * row.flatten()

# Invert alpha
alpha_inv = np.linalg.inv(alpha)

# Use alpha_inv,v to compute r-dependant h
for idx_r,rho2_r in enumerate(rho2):
	rho2_r       = rho2_r.flatten()
	h_r          = np.dot(alpha_inv, rho2_r-v)
	
	h[idx_r,...] = np.reshape(h_r, np.shape(rho2)[1:])

# Print h coefficients up to rank l_print
h_res      = np.zeros([n_bins,2])
h_res[:,0] = bins[:-1]

ctr        = 0

for l1 in range(l_print+1):
	for l2 in range(l_print+1):
		for l in range(l_print+1):
			if ( (l1 % 2 == 0) & (l2 % 2 == 0) & (l % 2 == 0) ):
				for m1 in range(-l1, l1+1):
					for m2 in range(-l2, l2+1):
						for m in range(-l, l+1):
							h_res[:,1] = h[:,a.sph_idx(l1,m1),a.sph_idx(l2,m2),a.sph_idx(l,m)]
							
							file_h = "%s/h_%d_%d%d_%d%d_%d%d.res" % (path,n_eq,l1,m1,l2,m2,l,m)
							np.savetxt(file_h, h_res)
							
							ctr += 1

print("\033[1;32m%d h coefficients printed in '%s/'\033[0m" % (ctr,path))
