from ovito.data import CutoffNeighborFinder
from ovito.modifiers import ExpressionSelectionModifier, DeleteSelectedModifier, WrapPeriodicImagesModifier
import numpy as np

# Taken from Soumendu's code
def cart2pol(X):
    x = X[0]
    y = X[1]
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)


def extract_local_angle(finder, index, central_atom_pos):

        neigh_index = []
        neigh_pos_polar = []
        neigh_pos_cart = []

        for neigh in finder.find(index):
                neigh_index.append(neigh.index)
                neigh_pos_cart.append(np.array([neigh.delta[0]+central_atom_pos[0], neigh.delta[1]+central_atom_pos[1]]))
                neigh_pos_polar.append(cart2pol(np.array([neigh.delta[0], neigh.delta[1]])))


        neigh_pos_polar = np.array(neigh_pos_polar)
        neigh_pos_cart = np.array(neigh_pos_cart)


        neigh_pos_polar_sorted = neigh_pos_polar[abs(neigh_pos_polar[:,1]).argsort()]
        neigh_pos_cart_sorted = neigh_pos_cart[abs(neigh_pos_polar[:,1]).argsort()]

        neigh_index = np.array(neigh_index)
        neigh_index_sorted = neigh_index[abs(neigh_pos_polar[:,1]).argsort()]

        is_zero = True # In case interested to implement local twists in future

        return {'index' :neigh_index_sorted[0], 'polar_angle': neigh_pos_polar_sorted[0,1], \
                'rel_cart_pos': neigh_pos_cart_sorted[0,:], 'zero_axis': is_zero}


def extract_layer_alignment(finder, initial_index, num_iter=10):
        ic = 0
        lattice_axis_info = {'index': [], 'polar_angle':[], \
                             'rel_cart_pos': [], \
                             'zero_axis': []}
        index = initial_index
        central_atom_pos= np.array([0,0])
        while ic < num_iter:
                local_angle_info = extract_local_angle(finder, index, central_atom_pos)

                # might be smarter way to do the following, for now bootstrap!
                lattice_axis_info['index'].append(local_angle_info['index'])
                lattice_axis_info['polar_angle'].append(local_angle_info['polar_angle'])
                lattice_axis_info['rel_cart_pos'].append(local_angle_info['rel_cart_pos'])
                lattice_axis_info['zero_axis'].append(local_angle_info['zero_axis'])

                index = local_angle_info['index']
                central_atom_pos = local_angle_info['rel_cart_pos']
                print (f'Done with iteration number: {ic}')
                ic+=1

        return lattice_axis_info



def compute_interlayer_angle(layer_1_info, layer_2_info):

        from scipy import stats
        from scipy.stats import t

        x1 = x2 = y1 = y2 = np.array([0.])

        x1=np.append(x1, np.array(layer_1_info['rel_cart_pos'])[:,0][layer_1_info['zero_axis']])
        y1=np.append(y1, np.array(layer_1_info['rel_cart_pos'])[:,1][layer_1_info['zero_axis']])

        x2=np.append(x2, np.array(layer_2_info['rel_cart_pos'])[:,0][layer_2_info['zero_axis']])
        y2=np.append(y2, np.array(layer_2_info['rel_cart_pos'])[:,1][layer_2_info['zero_axis']])

        res1=stats.linregress(x1,y1)
        res2=stats.linregress(x2,y2)

        # plt.plot(x1,y1, 'bo')
        # plt.plot(x1, res1.intercept + res1.slope*x1, 'b-')
        # plt.plot(x2,y2, 'ro')
        # plt.plot(x2, res2.intercept + res2.slope*x2, 'r-')

        tinv = lambda p, df: abs(stats.t.ppf(p/2, df))
        ts1 = tinv(0.05, len(x1)-2)
        err_1 = ts1*res1.stderr

        ts2 = tinv(0.05, len(x2)-2)
        err_2 = ts2*res2.stderr

        num_x = abs(res1.slope -res2.slope)
        denom_y = abs(1 + res1.slope*res2.slope)

        twist_angle = np.arctan2(num_x, denom_y)*180/np.pi

        return twist_angle, err_1, err_2

def get_interlayer_twist(data, cutoff, grid_resolution=1, reference_particle_type=2, num_iter=10):
        
	modifier_select = ExpressionSelectionModifier(expression=f"ParticleType=={reference_particle_type}")
	modifier_delete = DeleteSelectedModifier()
	data.data.apply(modifier_select)
	data.data.apply(modifier_delete)
	wrap_modifier = WrapPeriodicImagesModifier()
	data.data.apply(wrap_modifier)
			
	grid_centers_info = extract_layer_grid_center_ids(data, grid_resolution=grid_resolution)
	twist_angles = []

	for i in range(grid_resolution**2):
		id_1 = grid_centers_info['upper_layer_grid_center_ids'][i]['center_id']
		id_2 = grid_centers_info['lower_layer_grid_center_ids'][i]['center_id']

		index_1 = np.argwhere(data.data.particles["Particle Identifier"]==id_1)

		assert data.data.particles["Particle Identifier"][index_1] == id_1

		index_2 = np.argwhere(data.data.particles["Particle Identifier"]==id_2)
		assert data.data.particles["Particle Identifier"][index_1] == id_1

		finder = CutoffNeighborFinder(cutoff, data.data)
		layer_1_info = extract_layer_alignment(finder, index_1, num_iter)
		layer_2_info = extract_layer_alignment(finder, index_2, num_iter)
		twist_angles.append(compute_interlayer_angle(layer_1_info, layer_2_info)[0])
    
	return twist_angles

def extract_layer_grid_center_ids(data, metal_atom_type=1, grid_resolution=10, n_clusters=2):
    """
    Extract single center ID for each grid cell in layers
    """
    z_positions = data.data.particles.positions[...][:,2]
    # Cluster based on z-positions
    labels, _ = cluster_1d_vector(z_positions, n_clusters=n_clusters)
    
    def get_layer_data(layer_index):
        layer_mask = (labels == layer_index) & (data.data.particles.particle_types == metal_atom_type)
        layer_positions = data.data.particles.positions[...][layer_mask]
        layer_ids = data.data.particles.identifiers[...][layer_mask]
        return layer_positions, layer_ids
    
    # Extract layer positions and IDs
    upper_layer_pos, upper_layer_ids = get_layer_data(0)
    lower_layer_pos, lower_layer_ids = get_layer_data(1)
    
    def compute_grid_center_ids(positions, ids):
        x_bins = np.linspace(positions[:, 0].min(), positions[:, 0].max(), grid_resolution+1)
        y_bins = np.linspace(positions[:, 1].min(), positions[:, 1].max(), grid_resolution+1)
        
        grid_center_ids = []
        for i in range(grid_resolution):
            for j in range(grid_resolution):
                # Mask for current grid cell
                grid_mask = (
                    (positions[:, 0] >= x_bins[i]) & (positions[:, 0] < x_bins[i+1]) &
                    (positions[:, 1] >= y_bins[j]) & (positions[:, 1] < y_bins[j+1])
                )
                
                cell_positions = positions[grid_mask]
                cell_ids = ids[grid_mask]
                
                if len(cell_positions) > 0:
                    # Find center point and its corresponding ID
                    center_index = np.argmin(
                        np.sum((cell_positions - cell_positions.mean(axis=0))**2, axis=1)
                    )
                    center_id = cell_ids[center_index]
                    
                    grid_center = {
                        'x_range': (x_bins[i], x_bins[i+1]),
                        'y_range': (y_bins[j], y_bins[j+1]),
                        'center_id': center_id
                    }
                    grid_center_ids.append(grid_center)
        
        return grid_center_ids
    
    return {
        'upper_layer_grid_center_ids': compute_grid_center_ids(upper_layer_pos, upper_layer_ids),
        'lower_layer_grid_center_ids': compute_grid_center_ids(lower_layer_pos, lower_layer_ids)
    }


def find_nearest_center_point(points_x, points_y):
    """
    Find the point closest to the cluster center
    
    Args:
        points (array-like): List of (x,y) coordinates
    
    Returns:
        numpy.ndarray: Closest point to cluster center
    """
    from scipy.spatial.distance import cdist

    points_array = np.column_stack((points_x, points_y))
    
    # Calculate cluster center
    center = np.mean(points_array, axis=0)
    
    # Calculate distances from center
    distances = cdist(points_array, [center])
    
    # Find index of point closest to center
    nearest_point_index = np.argmin(distances)
    
    return points_array[nearest_point_index]


def cluster_1d_vector(data, n_clusters=3):
    """
    Cluster a 1D vector using K-means algorithm
    
    Parameters:
    data (array-like): 1D vector to be clustered
    n_clusters (int): Number of clusters (default: 3)
    
    Returns:
    tuple: (cluster labels, cluster centers)
    """
    from sklearn.cluster import KMeans
    # Reshape data for sklearn
    X = np.array(data).reshape(-1, 1)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    
    return kmeans.labels_, kmeans.cluster_centers_
