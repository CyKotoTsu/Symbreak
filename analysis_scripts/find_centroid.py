import numpy as np
import skimage as ski
from skimage import filters
from skimage import morphology
from skimage import measure
from scipy import fftpack, ndimage
import pandas as pd

def dve_to_val(p_mask):
    '''
    Docstring for dve_to_val
    Inputa:
        p_mask: (N) array of cell typa (0-2)
    Utputa:
        v: (N) binary values (1 if DVE, 0 otherwise)
    '''
    p_mask = np.asarray(p_mask,dtype=int)

    v = np.zeros(len(p_mask),dtype=int)
    v[p_mask==2] = 1
    return v


def points_to_mean_volume(points, values, morph, voxel_size=1.0, bounds=None):
    """
    points: (N, 3) array of xyz coordinates
    values: (N,) array of scalar values at each pointa
    morph: integer 0-3 with 0 as BMP, 1 as DKK, 2 as WNT and 3 as DVE
    voxel_size: size of a voxel in same units as points
    bounds: optional ((xmin,xmax),(ymin,ymax),(zmin,zmax)).
            If None, uses min/max of points.
    Returns:
      grid: 3D array (nx, ny, nz) with corresponding mean values where occupied
      origin: xyz coordinate of grid index (0,0,0)
    """
    points = np.asarray(points, dtype=float)
    values = np.asarray(values, dtype=float)
    if len(points) != len(values):
        raise ValueError("points and values must have same length")

    if bounds is None:
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
    else:
        mins = np.array([bounds[0][0], bounds[1][0], bounds[2][0]], dtype=float)
        maxs = np.array([bounds[0][1], bounds[1][1], bounds[2][1]], dtype=float)

    eps = 1e-9
    mins = mins - eps
    maxs = maxs + eps

    dims = np.ceil((maxs - mins) / voxel_size).astype(int)
    sum_grid = np.zeros(tuple(dims), dtype=float)
    cnt_grid = np.zeros(tuple(dims), dtype=np.int32)

    idx = np.floor((points - mins) / voxel_size).astype(int)
    valid = np.all((idx >= 0) & (idx < dims), axis=1)
    idx = idx[valid]

    if morph == 3:
        v = values[valid]
    else:
        v = values[:,morph][valid]

    np.add.at(sum_grid, (idx[:,0], idx[:,1], idx[:,2]), v)
    np.add.at(cnt_grid, (idx[:,0], idx[:,1], idx[:,2]), 1)

    mean_grid = np.zeros_like(sum_grid)
    mask = cnt_grid > 0
    mean_grid[mask] = sum_grid[mask] / cnt_grid[mask]
    return mins, dims, mean_grid, mins


def segmentation(grid,scale_thr):
    """
    grid: 3D array (nx, ny, nz) with scalar values
    Returns:
      df: dataframe with stuff
    """
    threshold = filters.threshold_otsu(grid)
    binary = grid > scale_thr*threshold
    label = measure.label(binary)
    regions = measure.regionprops_table(label,intensity_image=grid,properties=('area','bbox','centroid'))
    df = pd.DataFrame(regions)
    return df

def back_transform(voxel_coords, mins, voxel_size):
    return (voxel_coords+0.5)*voxel_size+mins

def extract_centroids(df, mins, voxel_size):
    centroids = []
    for i in range(len(df)):
        centroid = (df.loc[i,'centroid-0'], df.loc[i,'centroid-1'], df.loc[i,'centroid-2'])
        centroid = back_transform(np.array(centroid), mins, voxel_size)
        centroids.append(centroid)
    return np.array(centroids)


def points_to_centroid(points, values, morph, voxel_size=1.0, bounds=None, scale_thr=1.0):
    mins, dims, mean_grid, origin = points_to_mean_volume(points, values, morph, voxel_size, bounds)
    df = segmentation(mean_grid, scale_thr)
    centroids = extract_centroids(df, mins, voxel_size)
    return centroids