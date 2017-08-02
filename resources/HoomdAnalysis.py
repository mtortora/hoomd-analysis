#!/usr/bin/env python

import os
import sys

import gsd.pygsd
import gsd.hoomd

import numpy as np


# Trajectory analyser class
class Analyser():

	def __init__(self, file_traj):
		if not os.path.isfile(file_traj):
			print("\033[1;31mCouldn't find file '%s'\033[0m" % file_traj)
			sys.exit()

		data        = gsd.pygsd.GSDFile(open(file_traj, 'rb'))
		self.traj   = gsd.hoomd.HOOMDTrajectory(data)
		
		self.n_conf = len(self.traj)
		self.n_part = self.traj[0].particles.N


	# x,y,z box dimensions
	def box_dims(self, snap): return snap.configuration.box[:3]


	# Nematic order parameter/director from Q spectral analysis
	def ord_param(self, snap):
		qs = np.zeros([self.n_part,3,3])

		# Ensemble-averaged Q tensor
		for idx,quat in enumerate(snap.particles.orientation):
			u0 = 2 * (quat[1]*quat[3] + quat[0]*quat[2])
			u1 = 2 * (quat[2]*quat[3] - quat[0]*quat[1])
			u2 = 1 - 2.*quat[1]**2 - 2.*quat[2]**2
			
			qs[idx,0,0] = u0*u0
			qs[idx,1,0] = u1*u0
			qs[idx,2,0] = u2*u0
			qs[idx,0,1] = u0*u1
			qs[idx,1,1] = u1*u1
			qs[idx,2,1] = u2*u1
			qs[idx,0,2] = u0*u2
			qs[idx,1,2] = u1*u2
			qs[idx,2,2] = u2*u2
		
		q           = 3/2. * np.mean(qs, axis=0) - 1/2. * np.eye(3)
		evals,evecs = np.linalg.eigh(q)

		# Sort eigenvalues/vectors in decreasing order
		ordered     = evals.argsort()
	
		evals       = evals[ordered]
		evecs       = evecs[:,ordered]
	
		return evals[-1]


	# Vectorised pairwise distance computations
	def p_dists(self, snap):
		box       = snap.configuration.box[:3]
		positions = snap.particles.position
		
		# Avoid double-counting
		unique    = np.triu_indices(self.n_part, k=1)
		
		dists     = positions[:,None,...] - positions[None,...]
		dists     = dists[unique[0],unique[1],...]
		
		# Minimum image convention for PBCs
		for idx_axis in range(3):
			crossed_inf = (dists[...,idx_axis] < -box[idx_axis]/2.)
			crossed_sup = (dists[...,idx_axis] >  box[idx_axis]/2.)
			
			dists[crossed_inf,idx_axis] += box[idx_axis]
			dists[crossed_sup,idx_axis] -= box[idx_axis]
		
		dists = np.sqrt(np.sum(dists**2, axis=1))
		
		return dists


	# Accumulates class property 'prop' over the last n_eq configurations
	def accumulate(self, prop, n_eq, log=False):
		values = []
	
		if n_eq > self.n_conf: print("Warning: chosen equilibration time longer than trajectory")

		for idx,snap in enumerate(self.traj[-n_eq:]):
			if log:
				if (idx+1) % 100 == 0: print("\033[1;34mProcessed %d out of %d configurations\033[0m" % (idx+1, n_eq))

			values.append(prop(snap))

		return np.asfarray(values)


	# PDF from binned pairwise distances
	def g_hist(self, n_eq, n_bins=100, r_min=0.2, r_max=5):
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
		vol_bins  = 4 * np.pi * bins[:-1]**2 * dr

		hist      = np.asfarray(hist)
		
		hist     /= self.n_part*(self.n_part-1)/2. * n_eq
		hist     *= vol_ave/vol_bins

		return bins, hist
