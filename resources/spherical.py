#!/usr/bin/env python

import os
import sys

import numba

import numpy as np
import scipy.sparse as sps

from clebsch import cg_tabulate
from hankel  import hk_t, inv_hk_t

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

r_min   = 0.
r_max   = 12.

k_min   = 0.
k_max   = 4.

n_k     = 100
n_t     = 250

n_fit   = 5

l_print = min(2, l_max)


##################################
## Load HoomdAnalyser class     ##
##################################

path_traj = os.path.dirname(file_traj)
a         = Analyser(file_traj)

# Setup sparse solver
sps.linalg.use_solver(assumeSortedIndices=True, useUmfpack=False)

# Save data array up to rank l_print with corresponding x values
def print_f(x_vals, data, path, name):
	if not os.path.exists(path): os.makedirs(path)
	
	n_x      = x_vals.shape[0]
	res      = np.zeros([n_x,2])
	
	ctr      = 0
	res[:,0] = x_vals

	for idx in range(n_tot):
		l1,m1,l2,m2,l,m = inds[idx]
	
		if ( (l1 < l_print+1) & (l2 < l_print+1) & (l < l_print+1) ):
			res[:,1] = data[:,idx]
		
			file_res = "%s/%s_%d_%d_%d%d_%d%d_%d%d.res" % (path,name,n_eq,l_max,l1,m1,l2,m2,l,m)
			np.savetxt(file_res, res)
		
			ctr += 1

	print("\033[1;32m%d %s coefficients printed in '%s/'\033[0m" % (ctr,name,path))


##################################
## Compute harmonic indices     ##
##################################

inds = []

# Prune symmetrically-redundant indices
for l1 in range(l_max+1):
	if l1 % 2 == 0:
		for m1 in range(-l1, l1+1):
			for l2 in range(l1, l_max+1):
				if l2 % 2 == 0:
					for m2 in range(m1, l2+1):
						if m1 + m2 >= 0:
							for l in range(m1+m2, l_max+1):
								if l % 2 == 0:
									inds.append([l1,m1,l2,m2,l,-m1-m2])

inds  = np.asarray(inds, dtype=np.int32)
n_tot = inds.shape[0]

# Work out degeneracies
degs  = np.zeros(n_tot, dtype=np.int32)

for idx in range(n_tot):
	l1,m1,l2,m2,l,m = inds[idx]
	
	if m1 == m2 == 0:
		if l1 == l2: degs[idx] = 1
		else: degs[idx] = 2
	
	else:
		if ( (abs(m1) == abs(m2)) & (l1 == l2) ): degs[idx] = 2
		else: degs[idx] = 4

# Tabulate relevant Clebsch-Gordan coefficients
CGs = cg_tabulate(l_max)

print("\033[1;36mPrecomputed %d Clebsch-Gordan coefficients\033[0m" % np.size(CGs))


##################################
## Single-particle averages     ##
##################################

# Average single particle harmonics, order parameter and box volumes
f      = a.accumulate(a.single_sh_aves, n_eq, l_max=l_max)
dims   = a.accumulate(a.box_dims,  n_eq)

f      = f.mean(axis=0)

ops    = a.accumulate(a.nematic_q, n_eq)
op_ave = ops.mean()

vols   = dims[:,0]*dims[:,1]*dims[:,2]
rho    = (a.n_part/vols).mean()

# Compute orientation distribution function
t_bins = np.linspace(0, np.pi, num=n_t+1, dtype=np.float32)

ts     = t_bins[1:]
psis   = np.zeros(n_t+1, dtype=np.float32)

for idx_t,t in enumerate(t_bins):
	for l in range(l_max+1):
		if l % 2 == 0: psis[idx_t] += (f[l]*a.get_sph_harm(l, 0, t, 0)).real

print("Order parameter: %f, <Y20>: %f" % (op_ave, f[2]*np.sqrt(4*np.pi/5.)))

psi_res      = np.zeros([n_t+1,2], dtype=np.float32)

psi_res[:,0] = t_bins
psi_res[:,1] = psis

file_f       = "%s/f_%d_%d.res"   % (path_traj, n_eq, l_max)
file_psi     = "%s/psi_%d_%d.res" % (path_traj, n_eq, l_max)

np.savetxt(file_f, f)
np.savetxt(file_psi, psi_res)

print("\033[1;32mf's printed to '%s'\033[0m" % file_f)
print("\033[1;32mpsi printed to '%s'\033[0m" % file_psi)

# Gubbins equation
dts = np.diff(t_bins)

dps = np.diff(psis)/dts
dls = np.diff(np.log(np.abs(psis)))/dts

gs  = (rho*np.pi * dts * np.sin(ts) * dps*dls).sum()

print("\033[1;35mGubbins coefficient: %.3f\033[0m" % gs)


##################################
## Pair averaged correlations   ##
##################################

# Compute pair correlation function/spherical harmonics coefficients
r_bins,gr = a.g_hist(n_eq, n_bins=n_bins, r_min=r_min, r_max=r_max)

rho2      = a.average(a.pair_sh_aves, n_eq, bins=r_bins, inds=inds)
rho2     *= 4*np.pi*rho**2 * gr[:,None]

# Save rho2 coefficients up to rank l_print
path      = "%s/harmonics" % path_traj
rs        = r_bins[1:]

print_f(rs, rho2, path, 'rho')


##################################
## H-matrix inversion           ##
##################################

# Invert H-equation in the form rho2_r = alpha*h_r + v, with rho2_r,h_r,v of size n_tot
h = np.zeros_like(rho2, dtype=np.float32)

# Symmetrise rho2
for idx in range(n_tot):
	l1,m1,l2,m2,l,m = inds[idx]
	
	if ( (l1 == l2) & (abs(m1) == abs(m2)) ): rho2[:,idx] *= degs[idx]
	else:					                  rho2[:,idx] *= degs[idx]/2.

# Compute v
@numba.jit("void(f4[:],i4[:,:],f4[:],f4)",nopython=True)
def set_v_coeffs(_f, _inds, _v, rho):
	n_tot = _inds.shape[0]

	for idx in range(n_tot):
		l1,m1,l2,m2,l,m = _inds[idx]
	
		if ( (m1 == 0) & (m2 == 0) & (l == 0) & (m == 0) ):
			coeff   = rho**2 * np.sqrt(4*np.pi) * _f[l1]*_f[l2]
			_v[idx] = coeff

v = np.zeros(n_tot, dtype=np.float32)
set_v_coeffs(f, inds, v, rho)

# Partial f-weighted Clebsch-Gordan sums
@numba.jit("i4(i4,i4)",nopython=True)
def sph_idx(l, m): return int(l*(l+1)/2 + m)

@numba.jit("f4(f4[:,:,:],f4[:],i4,i4,i4,i4)",nopython=True)
def cg_1sum(_CGs, _f, l3, lp3, m3, l_max):
	c_sum = 0.

	if lp3 >= abs(m3):
		for lpp3 in range(l_max+1):
			if lpp3 % 2 == 0:
				coeff  = _CGs[sph_idx(l3,m3),sph_idx(lp3,m3),sph_idx(lpp3,0)] * _f[lpp3]
				c_sum += coeff

	return c_sum

@numba.jit("f4(f4[:,:,:],f4[:],i4,i4,i4,i4,i4,i4,i4)",nopython=True)
def cg_2sum(_CGs, _f, l1, l2, lp1, lp2, m1, m2, l_max):
	c_sum = 0.
	
	if ( (lp1 >= abs(m1)) & (lp2 >= abs(m2)) ):
		for lpp1 in range(l_max+1):
			for lpp2 in range(l_max+1):
				if ( (lpp1 % 2 == 0) & (lpp2 % 2 == 0) ):
					coeff1 = _CGs[sph_idx(l1,m1),sph_idx(lp1,m1),sph_idx(lpp1,0)] * _f[lpp1]
					coeff2 = _CGs[sph_idx(l2,m2),sph_idx(lp2,m2),sph_idx(lpp2,0)] * _f[lpp2]
	
					c_sum += coeff1*coeff2
	
	return c_sum

# Alpha coefficient setter
@numba.jit("void(f4[:,:,:],f4[:],i4[:,:],i4[:],i4[:],i4[:],f4[:],f4,i4)",nopython=True)
def set_alpha_coeffs(_CGs, _f, _inds, _degs, _rows, _cols, _data, rho, l_max):
	ctr   = 0
	n_tot = _inds.shape[0]

	for idx1 in range(n_tot):
		l1 = _inds[idx1,0]
		m1 = _inds[idx1,1]
		
		l2 = _inds[idx1,2]
		m2 = _inds[idx1,3]
		
		l  = _inds[idx1,4]
		m  = _inds[idx1,5]
	
		for idx2 in range(n_tot):
			lp1 = _inds[idx2,0]
			mp1 = _inds[idx2,1]
			
			lp2 = _inds[idx2,2]
			mp2 = _inds[idx2,3]
			
			lp  = _inds[idx2,4]
			mp  = _inds[idx2,5]
			
			if ( (mp1 == m1) & (mp2 == m2) & (lp == l) & (mp == m) ):
				coeff1     = cg_2sum(_CGs,_f,l1,l2,lp1,lp2,m1,m2,l_max)
				coeff2     = cg_2sum(_CGs,_f,l1,l2,lp2,lp1,m1,m2,l_max)
				
				coeff      = rho**2 * (coeff1+coeff2)/2. * _degs[idx2]

				_rows[ctr] = idx1
				_cols[ctr] = idx2
				_data[ctr] = coeff

				ctr       += 1

# Find number of non-zero alpha elements
inds_t   = np.delete(inds, [0,2], axis=-1)
n_coeffs = np.all((inds_t[:,None,:]==inds_t[None,:,:]), axis=-1).sum()

# Empty containers for sparse coo_matrix constructor
rows     = np.zeros(n_coeffs, dtype=np.int32)
cols     = np.zeros(n_coeffs, dtype=np.int32)
data     = np.zeros(n_coeffs, dtype=np.float32)

# Work-out alpha in sparse matrix format
set_alpha_coeffs(CGs, f, inds, degs, rows, cols, data, rho, l_max)
alpha = sps.coo_matrix((data, (rows,cols)), shape=(n_tot,n_tot), dtype=np.float32).tocsc()

# Invert alpha and solve for h
alpha_inv = sps.linalg.inv(alpha)

for idx_r,rho2_r in enumerate(rho2): h[idx_r,:] = alpha_inv.dot(rho2_r-v)

# Save h coefficients up to rank l_print
print_f(rs, h, path, 'h')


##################################
## Hankel transform             ##
##################################

drs    = np.diff(r_bins)

# Satisfy the Shannon-Nyquist criterium
dr_max = drs.max()
k_max  = min(k_max, np.pi/dr_max)

k_bins = np.linspace(k_min**(1/2.), k_max**(1/2.), num=n_k+1, dtype=np.float32)**2

ks     = k_bins[1:]
hk     = hk_t(h, inds, r_bins, ks)

# Save hk coefficients up to rank l_print
print_f(ks, hk, path, 'hk')


##################################
## Ornstein-Zernike (OZ) solver ##
##################################

# Invert Fourier-space OZ in the form h_k = beta_k*c_k with h_k,c_k of size n_tot
ck  = np.zeros_like(hk, dtype=np.float32)

# Symmetrise h_k for lhs term
hks = hk.copy()

for idx in range(n_tot):
	l1,m1,l2,m2,l,m = inds[idx]
	
	if ( (l1 == l2) & (abs(m1) == abs(m2)) ): hks[:,idx] *= degs[idx]
	else:					                  hks[:,idx] *= degs[idx]/2.

# Symmetrised convenience function for non-diagonal beta_k coefficients
@numba.jit("f4(f4[:,:,:],f4[:],f4[:],i4[:,:],i4[:],i4,i4,i4,i4,i4,i4,i4,i4,i4)",nopython=True)
def zeta(_CGs, _hk, _f, _inds, _degs, l3, m3, l2, m2, l, m, lp, mp, l_max):
	c_sum = 0.
	n_tot = _inds.shape[0]

	for idx in range(n_tot):
		lp3   = _inds[idx,0]
		mp3   = _inds[idx,1]
		
		lp2   = _inds[idx,2]
		mp2   = _inds[idx,3]
		
		lpp   = _inds[idx,4]
		mpp   = _inds[idx,5]
		
		cond1 = ( (mp3 == m3) & (lp2 == l2) & (mp2 == m2) )
		cond2 = ( (mp2 == m3) & (lp3 == l2) & (mp3 == m2) )
		
		if ( cond1 | cond2 ):
			coeff = _CGs[sph_idx(l,m),sph_idx(lp,mp),sph_idx(lpp,mpp)]

			if cond1: c_sum += coeff * cg_1sum(_CGs,_f,l3,lp3,m3,l_max) * _hk[idx]*_degs[idx]
			if cond2: c_sum += coeff * cg_1sum(_CGs,_f,l3,lp2,m3,l_max) * _hk[idx]*_degs[idx]

	return (-1.)**m3 * c_sum/4.

# Beta coefficient setter
@numba.jit("void(f4[:,:,:],f4[:],f4[:],i4[:,:],i4[:],i4[:],i4[:],f4[:],f4,i4)",nopython=True)
def set_beta_coeffs(_CGs, _hk, _f, _inds, _degs, _rows, _cols, _data, rho, l_max):
	ctr   = 0
	n_tot = _inds.shape[0]
	
	for idx1 in range(n_tot):
		l1 = _inds[idx1,0]
		m1 = _inds[idx1,1]
		
		l2 = _inds[idx1,2]
		m2 = _inds[idx1,3]
		
		l  = _inds[idx1,4]
		m  = _inds[idx1,5]
		
		for idx2 in range(n_tot):
			lp1   = _inds[idx2,0]
			mp1   = _inds[idx2,1]
			
			lp2   = _inds[idx2,2]
			mp2   = _inds[idx2,3]
			
			lp    = _inds[idx2,4]
			mp    = _inds[idx2,5]
			
			cond1 = ( (lp1 == l1) & (mp1 == m1) )
			cond2 = ( (lp2 == l1) & (mp2 == m1) )
			
			if ( cond1 | cond2 ):
				coeff = 0.

				# Symmetrised h_k/c_k coupling coefficients
				if cond1:
					cz1    = zeta(_CGs,_hk,_f,_inds,_degs,lp2,mp2,l2,m2,l,m,lp,mp,l_max)
					coeff += rho*cz1 * _degs[idx2]
		
				if cond2:
					cz2    = zeta(_CGs,_hk,_f,_inds,_degs,lp1,mp1,l2,m2,l,m,lp,mp,l_max)
					coeff += rho*cz2 * _degs[idx2]
			
				# Identity matrix with symmetrised degeneracy for decoupled rhs c_k
				if ( cond1 & (lp2 == l2) & (mp2 == m2) & (lp == l) & (mp == m) ):
					if ( (l1 == l2) & (abs(m1) == abs(m2)) ): coeff += 1. * degs[idx2]
					else:				                      coeff += 1. * degs[idx2]/2.
				
				_rows[ctr] = idx1
				_cols[ctr] = idx2
				_data[ctr] = coeff
				
				ctr       += 1

# Find upper bound for number of non-zero beta elements
inds_t1   = np.delete(inds, [2,3,4,5], axis=-1)
inds_t2   = np.delete(inds, [0,1,4,5], axis=-1)

n_coeffs1 = np.all((inds_t1[:,None,:]==inds_t1[None,:,:]), axis=-1).sum()
n_coeffs2 = np.all((inds_t1[:,None,:]==inds_t2[None,:,:]), axis=-1).sum()

n_coeffs  = n_coeffs1+n_coeffs2
				
# Empty containers for sparse coo_matrix constructor
rows      = np.zeros(n_coeffs, dtype=np.int32)
cols      = np.zeros(n_coeffs, dtype=np.int32)
data      = np.zeros(n_coeffs, dtype=np.float32)

for idx_k in range(n_k):
	h_k  = hk[idx_k,:]
	h_ks = hks[idx_k,:]
	
	# Work-out beta in sparse matrix format
	set_beta_coeffs(CGs, h_k, f, inds, degs, rows, cols, data, rho, l_max)
	beta = sps.coo_matrix((data, (rows,cols)), shape=(n_tot,n_tot), dtype=np.float32).tocsc()

	# Solve for ck using symmetrised hks for lhs
	ck[idx_k,:] = sps.linalg.spsolve(beta, h_ks)

	print("\033[1;36mSuccessfully resolved %d out of %d iso-k surfaces\033[0m" % (idx_k+1,n_k))

# Save ck coefficients up to rank l_print
print_f(ks, ck, path, 'ck')


##################################
## Inverse Hankel transforms    ##
##################################

c     = inv_hk_t(ck, inds, k_bins, rs)
h_inv = inv_hk_t(hk, inds, k_bins, rs)

# Save coefficients up to rank l_print
print_f(rs, c,     path, 'c')
print_f(rs, h_inv, path, 'h_inv')


##################################
## Frank elastic constants      ##
##################################

cii = np.zeros([n_k,3], dtype=np.float32)

vi  = np.asarray([-1,-1,2], dtype=np.float32)
wi  = np.asarray([-1, 1,0], dtype=np.float32)

# Work out cii coefficients
for idx_k in range(n_k):
	for idx in range(n_tot):
		l1,m1,l2,m2,l,m = inds[idx]
			
		if ( (abs(m1) == 1) & (abs(m2) == 1) ):
			c_sum = np.zeros(3, dtype=np.float32)
			pref  = np.sqrt(l1*(l1+1.)*l2*(l2+1.)) * f[l1]*f[l2]

			if ( (l == 0) ):
				c_sum += ck[idx_k,idx]*degs[idx]

			if ( (l == 2) & (m == 0) ):
				c_sum += vi * np.sqrt(5)/2.  * ck[idx_k,idx]*degs[idx]

			if ( (l == 2) & (abs(m) == 2) ):
				c_sum += wi * np.sqrt(15/8.) * ck[idx_k,idx]*degs[idx]

			cii[idx_k,:] += rho**2/(8*np.sqrt(np.pi)) * pref*c_sum

# Save cii's and ki's
c_res      = np.zeros([n_k,3], dtype=np.float32)
c_res[:,0] = ks

ki         = np.zeros(3, dtype=np.float32)

for i in range(3):
	# Quadratic initial fit over n_fit points for ki's
	coeffs     = np.polyfit(ks[:n_fit]**2, cii[:n_fit,i], 1)
	fit        = np.poly1d(coeffs)(ks**2)
	
	ki[i]      = coeffs[0]

	c_res[:,1] = cii[:,i]
	c_res[:,2] = fit
	
	file_res   = "%s/c%d_%d_%d.res" % (path,i+1,n_eq,l_max)
	np.savetxt(file_res, c_res)

	print("\033[1;32mPrinted C%d to '%s'\033[0m" % (i+1,file_res))

file_ki = "%s/ks_%d_%d_%d.res" % (path,n_eq,l_max,n_fit)
np.savetxt(file_ki, ki)

print("\033[1;32mBest-fit Ks printed to '%s'\033[0m" % file_ki)
print("K1: %.3f, K2: %.3f, K3: %.3f (fitted over %d points)" % (ki[0],ki[1],ki[2],n_fit))
