import ovito
import numpy as np

from ovito.vis  import ParticleDisplay
from ovito.data import ParticleProperty

from matplotlib.cm import get_cmap


# ========================
# Particle parameters
# ========================

# Spherocylinder diameter and length
D    = 1.
L    = 12.

# Colormap for order parameters
cmap = "jet"


# ========================
# Setup initial state
# ========================

node                 = ovito.dataset.selected_node

part_disp            = node.source.particle_properties.position.display
part_disp.shape      = ParticleDisplay.Shape.Spherocylinder

# Needed by Ovito to properly handle aspherical particles from .gsd files
asym_prop            = node.source.create_particle_property(ParticleProperty.Type.AsphericalShape)
color_prop           = node.source.create_particle_property(ParticleProperty.Type.Color)
orient_prop          = node.source.create_particle_property(ParticleProperty.Type.Orientation)

# Initialise display parameters
asym_prop .marray[:] = [D/2.,0,L]
color_prop.marray[:] = [1.,0.,0.]

cmap                 = get_cmap(cmap)


# ========================
# Custom pipeline modifier
# ========================

def modify(frame, input, output):
	
	# Properties to be recomputed at every frame
	asym_prop    = output.create_particle_property(ParticleProperty.Type.AsphericalShape)
	color_prop   = output.create_particle_property(ParticleProperty.Type.Color)
	order_prop   = output.create_user_particle_property('OrderParam', "float")

	# Load input variables
	size         = output.number_of_particles
	quats        = output.particle_properties.orientation.array
	
	# Main particle axes from quaternions in the x,y,z,w convention
	u0           = 2 * (quats[:,2]*quats[:,0] + quats[:,3]*quats[:,1])
	u1           = 2 * (quats[:,2]*quats[:,1] - quats[:,3]*quats[:,0])
	u2           = 1 - 2.*quats[:,0]**2 - 2.*quats[:,1]**2

	qs           = np.zeros([size,3,3])
	
	# Compute molecular order-parameter tensor qs
	qs[:,0,0]    = u0*u0
	qs[:,1,0]    = u1*u0
	qs[:,2,0]    = u2*u0
	qs[:,0,1]    = u0*u1
	qs[:,1,1]    = u1*u1
	qs[:,2,1]    = u2*u1
	qs[:,0,2]    = u0*u2
	qs[:,1,2]    = u1*u2
	qs[:,2,2]    = u2*u2

	# Spectral analysis of the ensemble-averaged tensor q
	q            = 3/2. * np.mean(qs, axis=0) - 1/2. * np.eye(3)
	evals, evecs = np.linalg.eigh(q)

	# Reorder evals, evecs by decreasing eigenvalues
	idx          = evals.argsort()[::-1]
	
	evals        = evals[idx]
	evecs        = evecs[:, idx]
	
	# Average order parameter/director s and dir are the largest eval/evec of q
	s            = evals[0]
	dir          = evecs[:, 0]

	# order_params is the projection of long molecular axes on dir
	order_params = dir[0]*u0 + dir[1]*u1 + dir[2]*u2
	order_params = np.abs(order_params)

	# Color particles according to order_params rescaled by s
	colors       = cmap(s*order_params)[:,:3]
	
	# Update particle display properties
	asym_prop .marray[:] = [D/2., 0, L]
	color_prop.marray[:] = colors

	# Update order parameter
	order_prop.marray[:] = order_params
