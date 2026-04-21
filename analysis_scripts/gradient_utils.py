import numpy as np
import pandas as pd
import pickle
import matplotlib.tri as tri
import find_centroid as ppt

# Extract contrast
def contrast(file, morph, frame):
    ''' Extract contrast from simulation results
    Inputs:
        file - path to file of simulation results
        morph - morphogen to extract contrast for (BMP, DKK, Wnt)
        frame - frame number to extract contrast from
    Returns:
        contrast - max/mean'''
    with open(file,'rb') as f:
        p_mask_lst, x_lst, p_lst, q_lst, U_lst = pickle.load(f)
    
    U = U_lst[frame]

    U_max = max(U[:,morph])
    U_mean = np.mean(U[:,morph])
    contrast = U_max/U_mean
    return contrast

# Transform into angular coords
def angular(file,frame):
    ''' Transform cell positions into angular coordinates
    Inputs:
        file - path to file of simulation results
        frame - frame number to extract contrast from
    Returns:
        np.array of angular coordinates for each cell'''
    with open(file,'rb') as f:
        p_mask_lst, x_lst, p_lst, q_lst, U_lst = pickle.load(f)

    x = x_lst[frame]
    theta = np.arctan2(x[:,1], x[:,0])
    U = U_lst[frame]
    mask = p_mask_lst[frame]
    return [theta, x[:,2], U, mask]   

# Interpolate as contour map
def contour(angular_coords, morph, resolution):
    ''' Interpolate morphogen levels as contour map
    Inputs:
        angular_coords - angular coordinates
        morph - morphogen to extract contrast for (BMP, DKK, Wnt)
        resolution - number of points along each axis
    Returns:
        np.array of interpolated morphogen levels at each point on grid'''
    
    theta,z, U, mask = angular_coords

    xmin, xmax = -np.pi, np.pi
    zmin, zmax = min(z), max(z)
    theta_grid = np.linspace(xmin, xmax, resolution)
    z_grid = np.linspace(zmin, zmax, resolution)
  
  # Linearly interpolate the data (x, y) on a grid defined by (xi, yi).
    triang = tri.Triangulation(theta, z)
    interpolator = tri.LinearTriInterpolator(triang, U[:, morph])
    Xi, Yi = np.meshgrid(theta_grid, z_grid)
    zi = interpolator(Xi, Yi)
    
    return theta_grid, z_grid, zi

def get_width(file, frame, morph, resolution):
    '''
    Get width of morphogen gradient at half maximum concentration
    Inputs:
        file - path to file of simulation results
        frame - frame number to extract contrast from
        morph - morphogen to extract contrast for (BMP, DKK, Wnt)
        resolution - number of points along each axis for interpolation
    Returns:
        width - width of morphogen gradient at half maximum concentration
    '''
    angular_coords = angular(file, frame)
    theta_grid, z_grid, zi = contour(angular_coords, morph, resolution)
    half_max = np.max(zi) / 2
    max_idx = np.unravel_index(np.argmax(zi), zi.shape)
    # Get the line of concentration values along the x-axis at the y-coordinate of the maximum concentration
    line_conc = zi[max_idx[0], :]
    # Find where the concentration falls to half max on either side of the maximum concentration
    left_half_max_idx = np.where(line_conc[:max_idx[1]] < half_max)[0]
    right_half_max_idx = np.where(line_conc[max_idx[1]:] < half_max)[0] + max_idx[1]
    if len(left_half_max_idx) > 0 and len(right_half_max_idx) > 0:
        left_half_max_x = theta_grid[left_half_max_idx[-1]]
        right_half_max_x = theta_grid[right_half_max_idx[0]]
        width = right_half_max_x - left_half_max_x
        return width
    else:
        return None



# Transform frame to A-P gradient

def spin_cells(x,U):
    '''
    Rotate cells to the anterior side! With passion!
    Input: 
        x: 3D array of positions
        U: concentration matrix
    Returns: rotated x_array
    '''
    centroid = ppt.points_to_centroid(points = x, values = U[:,2], morph = 3,
                                      voxel_size=2.5, bounds = None,
                                      scale_thr = 1.5)[0]
    # rotate all avg_pos by -angle to align with y-axis using complexia numbers
    angle = np.arctan2(centroid[0], centroid[1])  # Angle in radians
    x_spin = x.copy()
    for i in range(len(x)):

        # Convert to complex number for rotation
        complex_pos = x[i][0] + 1j * x[i][1]
        rotated_complex = complex_pos * np.exp(1j * angle)
        x_spin[i] = np.array([rotated_complex.real, rotated_complex.imag, x[i][2]])
    return(x_spin)
        


# def find_anterior(x,mask):
#     '''
#     find the anteriro side!
#     Input:
#         x: 3D array of positions
#         mask: p_mask of cell types
    
#     Returns:

#     '''

#     # Find centroid and define A-P
#     values = ppt.dve_to_val(mask)
#     centroid = ppt.points_to_centroid(points = x, values = values, morph = 3,
#                                       voxel_size=2.5, bounds = None,
#                                       scale_thr = 1.5)[0]
    
#     div_line_slope = -(centroid[0]/centroid[1])
#     if centroid[1]>0:
#         ant_mask = x[:,1] > div_line_slope * x[:,0]
#         post_mask = x[:,1] <= div_line_slope * x[:,0]
#     else:
#         ant_mask = x[:,1] < div_line_slope * x[:,0]
#         post_mask = x[:,1] >= div_line_slope * x[:,0]

#     ant_idx = np.where(ant_mask)[0]
#     post_idx = np.where(post_mask)[0]

#     return ant_idx, post_idx

# def ap_axis(file, frame, morph, res_z):

#     '''
#     Input:
#         file:  Path to input file
#         frame: frame to read
#         morph: morphogen to extract (0~2)
#         res_z (int):  how well refined is the gradient grid

#     Output: 
#         dictionary of gradient values
#     '''
#     with open (file, 'rb') as f:
#         p_mask_lst, x_lst, _, _, U_lst = pickle.load(f)

#     x = x_lst[frame]
#     U = U_lst[frame]
#     msk = p_mask_lst[frame]

#     min_z = np.min(x[:,2])
#     spacing = -min_z/res_z
#     stack_marks = np.arange(min_z,0+spacing,spacing)

#     ant_idx, post_idx = find_anterior(x,msk)
#     ant_x = x[ant_idx]
#     post_x = x[post_idx]
#     ant_U = U[ant_idx]
#     post_U = U[post_idx]

#     plist = []
#     alist = []
#     # Sort anterior cells into a-p sections
#     for i in range(len(stack_marks)-1):
#         z_smol = stack_marks[i]
#         z_bik = stack_marks[i+1]
#         ant_mask = (z_smol < ant_x[:,2]) & (ant_x[:,2] < z_bik)
#         post_mask = (z_smol < post_x[:,2]) & (post_x[:,2] < z_bik)
        
#         ant_slice = ant_U[ant_mask]
#         post_slice = post_U[post_mask]

#         avg_c_ant = np.mean(ant_slice[:,morph])
#         avg_c_post = np.mean(post_slice[:,morph])
#         alist.append(avg_c_ant)
#         plist.append(avg_c_post)

#     # reverse plist 
#     plist_fin = plist[::-1]

#     ap_grad = {'values':alist+plist_fin}
#     return ap_grad

def quadrant_cells(x_spin):
    ant_mask = x_spin[:,1]>np.abs(x_spin[:,0])
    post_mask = x_spin[:,1]<-np.abs(x_spin[:,0])

    ant_idx = np.where(ant_mask)[0]
    post_idx = np.where(post_mask)[0]

    return ant_idx, post_idx



def ap_axis(file, frame, morph, res_z):
    '''
    Input:
         file:  Path to input file
         frame: frame to read
         morph: morphogen to extract (0~2)
         res_z (int):  how well refined is the gradient grid

     Output: 
         dictionary of gradient values
    '''
    with open (file, 'rb') as f:
        _, x_lst, _, _, U_lst = pickle.load(f)

    x = x_lst[frame]
    U = U_lst[frame]

    min_z = np.min(x[:,2])
    stack_marks = np.linspace(min_z, 0.0, res_z + 1)
    x_spin = spin_cells(x,U)
    ant_idx, post_idx = quadrant_cells(x_spin)
    ant_x = x[ant_idx]
    post_x = x[post_idx]
    ant_U = U[ant_idx]
    post_U = U[post_idx]

    plist = []
    alist = []
    # Sort anterior cells into a-p sections
    for i in range(len(stack_marks)-1):
        z_smol = stack_marks[i]
        z_bik = stack_marks[i+1]
        ant_mask = (z_smol < ant_x[:,2]) & (ant_x[:,2] < z_bik)
        post_mask = (z_smol < post_x[:,2]) & (post_x[:,2] < z_bik)
        
        ant_slice = ant_U[ant_mask]
        post_slice = post_U[post_mask]

        avg_c_ant = np.mean(ant_slice[:,morph]) if len(ant_slice) > 0 else np.nan
        avg_c_post = np.mean(post_slice[:,morph]) if len(post_slice) > 0 else np.nan
        alist.append(avg_c_ant)
        plist.append(avg_c_post)

    # reverse plist 
    alist_fin = alist[::-1]

    ap_grad = {'values':alist_fin+plist}
    return ap_grad







    
    


