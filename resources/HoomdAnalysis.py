#!/usr/bin/env python

import os
import sys

import gsd.pygsd
import gsd.hoomd

import numpy as np

from scipy.special import sph_harm


# Trajectory analyser class
class Analyser():

	def __init__(self, file_traj):
		if not os.path.isfile(file_traj):
			print("\033[1;31mCouldn't find file '%s'\033[0m" % file_traj)
			sys.exit()

		data            = gsd.pygsd.GSDFile(open(file_traj, 'rb'))
		self.traj       = gsd.hoomd.HOOMDTrajectory(data)
		
		self.n_conf     = len(self.traj)
		self.n_part     = self.traj[0].particles.N


	# x,y,z box dimensions
	def box_dims(self, snap): return snap.configuration.box[:3]
	
	
	# Main particle axes from quaternions
	def part_axis(self, quats):
		u0 = 2 * (quats[...,1]*quats[...,3] + quats[...,0]*quats[...,2])
		u1 = 2 * (quats[...,2]*quats[...,3] - quats[...,0]*quats[...,1])
		u2 = 1 - 2.*quats[...,1]**2 - 2.*quats[...,2]**2
	
		return np.asarray([u0,u1,u2]).T


	# Fetch spherical harmonic of indices l,m
	def get_sph_harm(self, l, m, theta, phi):
		if abs(m) > l: raise ValueError("Must have |m| <= l")
			
		return sph_harm(m, l, phi, theta)


	# Fetch all harmonics up to rank l_max
	def get_sph_harms(self, l_max, theta, phi):
		shs = np.zeros((l_max+1)**2, dtype=np.complex64)
			
		for l in range(l_max+1):
			if l % 2 == 0:
				for m in range(l+1):
					lmp = l*(l+1) + m
					lmn = l*(l+1) - m

					slm = sph_harm(m, l, phi, theta)

					shs[lmp] = slm
					shs[lmn] = (-1)**m * slm.conj()

		return shs


	# Project 3xn vector(s) vecs in the frame of 3x3 matrix rot
	def proj_vec(self, vecs, rot): return np.dot(rot.T, vecs.T).T
	
	
	# Vectorial polar/azimuthal angles
	def sph_angs(self, vecs):
		vecs  /= np.linalg.norm(vecs, axis=-1, keepdims=True)
	
		thetas = np.asarray(np.arccos(vecs[...,2]))
		phis   = np.asarray(np.arctan(vecs[...,1]/vecs[...,0]))
	
		# Handle azimuthal phi degeneracy
		phis[vecs[...,0] < 0] += np.pi
	
		return thetas,phis


	# Nematic director from Q spectral analysis
	def nematic_q(self, snap, mode="ord"):
		qs          = np.zeros([self.n_part,3,3])
		quats       = snap.particles.orientation

		# Ensemble-averaged Q tensor
		u0, u1, u2  = self.part_axis(quats).T
		
		qs[:,0,0]   = u0*u0
		qs[:,1,0]   = u1*u0
		qs[:,2,0]   = u2*u0
		qs[:,0,1]   = u0*u1
		qs[:,1,1]   = u1*u1
		qs[:,2,1]   = u2*u1
		qs[:,0,2]   = u0*u2
		qs[:,1,2]   = u1*u2
		qs[:,2,2]   = u2*u2
		
		q           = 3/2. * np.mean(qs, axis=0) - 1/2. * np.eye(3)
		evals,evecs = np.linalg.eigh(q)

		# Sort eigenvalues/vectors in increasing order
		ordered     = evals.argsort()
	
		evals       = evals[ordered]
		evecs       = evecs[:,ordered]
		
		if np.linalg.det(evecs) < 0: evecs[:,0] *= -1
	
		if mode == "ord":   return evals[-1]
		if mode == "frame": return evecs


	# Vectorised pairwise distance computations
	def p_dists(self, snap):
		box       = snap.configuration.box[:3]
		positions = snap.particles.position
		
		# Avoid double-counting
		unique    = np.triu_indices(self.n_part, k=1)
		
		vecs      = positions[:,None,...] - positions[None,...]
		vecs      = vecs[unique[0],unique[1],...]
		
		# Minimum image convention for PBCs
		for idx_axis in range(3):
			crossed_inf = (vecs[...,idx_axis] < -box[idx_axis]/2)
			crossed_sup = (vecs[...,idx_axis] >  box[idx_axis]/2)
			
			vecs[crossed_inf,idx_axis] += box[idx_axis]
			vecs[crossed_sup,idx_axis] -= box[idx_axis]
		
		dists = np.sqrt(np.sum(vecs**2, axis=1))
		
		return dists


	# Accumulates class property 'prop' over the last n_eq configurations
	def accumulate(self, prop, n_eq, log=False, **kwargs):
		if n_eq > self.n_conf: print("Warning: chosen equilibration time longer than trajectory")
		
		values = []
	
		for idx,snap in enumerate(self.traj[-n_eq:]):
			values.append(prop(snap, **kwargs))

			if log:
				if (idx+1) % 100 == 0: print("\033[1;34mProcessed %d out of %d configurations\033[0m"
											 % (idx+1, n_eq))

		return np.asarray(values)


	# Averages class property 'prop' over the last n_eq configurations
	def average(self, prop, n_eq, log=False, **kwargs):
		if n_eq > self.n_conf: print("Warning: chosen equilibration time longer than trajectory")

		iter_traj = iter(self.traj[-n_eq:])
		init_snap = next(iter_traj)
		
		sum       = prop(init_snap, **kwargs)
		
		for idx,snap in enumerate(iter_traj):
			if log: print("\033[1;34mProcessed %d out of %d configurations\033[0m"
						  % (idx+1, n_eq))
	
			sum += prop(snap, **kwargs)

		return sum/n_eq


	# PDF from binned pairwise distances
	def g_hist(self, n_eq, n_bins=300, r_min=0, r_max=12):
		dims      = self.accumulate(self.box_dims, n_eq)
		dists     = self.accumulate(self.p_dists,  n_eq, log=True)

		# Discard distances greater than r_max/smallest box dimension
		max_dist  = np.min(dims)/2.
		r_max     = min(max_dist, r_max)
		
		dists     = dists[dists<r_max]

		vols      = dims[:,0]*dims[:,1]*dims[:,2]
		vol_ave   = np.mean(vols)

		hist,bins = np.histogram(dists, bins=np.linspace(r_min,r_max,num=n_bins+1))

		# Renormalise histogram
		dr        = np.diff(bins)
		vol_bins  = 4 * np.pi * bins[1:]**2 * dr

		hist      = np.asfarray(hist)
		
		hist     /= self.n_part*(self.n_part-1)/2. * n_eq
		hist     *= vol_ave/vol_bins

		return bins,hist


	# Single-particle spherical harmonic averages
	def single_sph_aves(self, snap, l_max=8):
		quats         = snap.particles.orientation
		sph_sums      = np.zeros(l_max+1, dtype=np.complex64)

		# Project particle axes in nematic frame and fetch spherical angles
		part_axes     = self.part_axis(quats)
		frame         = self.nematic_q(snap, mode="frame")

		part_axes_prj = self.proj_vec(part_axes, frame)
		thetas,phis   = self.sph_angs(part_axes_prj)
		
		for theta,phi in zip(thetas,phis):
			for l in range(l_max+1):
				if l % 2 == 0: sph_sums[l] += self.get_sph_harm(l, 0, theta, phi)

		sph_aves = sph_sums / self.n_part

		return sph_aves


	# Pair spherical harmonic averages
	def pair_sph_aves(self, snap, bins=np.linspace(0, 12, num=300+1), l_max=8):
		n_bins        = len(bins)-1
		r_min,r_max   = np.min(bins),np.max(bins)
		
		box           = snap.configuration.box[:3]

		cm_pos        = snap.particles.position
		quats         = snap.particles.orientation
		
		n_harm        = (l_max+1)**2
		sph_sums      = np.zeros([n_bins,n_harm,n_harm,n_harm], dtype=np.complex64)
		
		p_ctr         = np.zeros(n_bins)
		
		# Project particle axes in nematic frame and fetch spherical angles
		part_axes     = self.part_axis(quats)
		frame         = self.nematic_q(snap, mode="frame")
		
		part_axes_prj = self.proj_vec(part_axes, frame)
		thetas,phis   = self.sph_angs(part_axes_prj)

		for idx_part1 in range(self.n_part):
			for idx_part2 in range(idx_part1+1, self.n_part):
				vec = cm_pos[idx_part2,:] - cm_pos[idx_part1,:]

				# Minimum image convention for PBCs
				for idx_axis in range(3):
					if vec[idx_axis] < -box[idx_axis]/2: vec[idx_axis] += box[idx_axis]
					if vec[idx_axis] >  box[idx_axis]/2: vec[idx_axis] -= box[idx_axis]
		
				r = np.linalg.norm(vec)

				if r_min < r < r_max:
					idx_r         = np.digitize(r, bins)-1
					p_ctr[idx_r] += 1
					
					vec_prj       = self.proj_vec(vec, frame)
					theta,phi     = self.sph_angs(vec_prj)
				
					theta1,phi1   = thetas[idx_part1],phis[idx_part1]
					theta2,phi2   = thetas[idx_part2],phis[idx_part2]
		
					shs           = self.get_sph_harms(l_max, theta, phi)
					shs1          = self.get_sph_harms(l_max, theta1, phi1)
					shs2          = self.get_sph_harms(l_max, theta2, phi2)
					
					sph_sums[idx_r,...] += np.einsum("i,j,k", shs, shs1, shs2)
	
		p_ctr[p_ctr==0] = 1
	
		sph_aves = sph_sums.conj() / p_ctr[:,None,None,None]
		
		return sph_aves
