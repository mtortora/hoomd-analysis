#!/usr/bin/env python

import os
import sys

import numpy as np
import HoomdAnalysis as ha


if len(sys.argv) != 4:
	print("\033[1;31mUsage is %s trajectory n_eq n_bins\033[0m" % sys.argv[0])
	sys.exit()


file_traj = os.path.realpath(sys.argv[1])

n_eq      = int(sys.argv[2])
n_bins    = int(sys.argv[3])

path_traj = os.path.dirname(file_traj)
a         = ha.Analyser(file_traj)

bins,hist = a.g_hist(n_eq, n_bins=n_bins)

gr        = np.zeros([n_bins, 2])

gr[:,0]   = bins[:-1]
gr[:,1]   = hist

file_g    = "%s/gr_%d_%d.res" % (path_traj, n_eq, n_bins)


np.savetxt(file_g, gr)

print("\033[1;32mg(r) printed to '%s'\033[0m" % file_g)
