 ### Imports ###
import matplotlib
import numpy as np
import torch
from scipy.spatial import cKDTree, Voronoi, voronoi_plot_2d
import os
import itertools
import pickle
from time import time
import json
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import seaborn as sns
import matplotlib.colors as colors
from matplotlib.patches import Polygon as MplPolygon

import matplotlib.cm as cm
from shapely.geometry import Point, Polygon
import pandas as pd

import napari
from qtpy.QtCore import QTimer
from napari_animation import Animation



### Track cells ###
def track_bottom_cells(file, threshold, plt_lim):
    ''' Choose bottom cells below a z-threshold and track these throughout our simulation
    Inputs: 
        file - path to file of simulation results
        threshold - z directional coordinate under which cells are tracked
        plt_lim - limit for plot axes
        
    Returns:
        2D plot of tracked cells over time'''
    
    with open(file , 'rb') as f:
        p_mask_lst, x_lst, p_lst, q_lst, U_lst = pickle.load(f)
    
    x_init = x_lst[0]
    tracked_idx = np.where(x_init[:,2] < threshold)[0]

    # Make 3 subplots
    fig, axs = plt.subplots(1, 4, figsize=(30, 6))
    for ax in axs:
        ax.set_xlim([plt_lim[0], plt_lim[1]])
        ax.set_ylim([plt_lim[2], plt_lim[3]])

    for i in range(len(tracked_idx)):
        cell = tracked_idx[i]
        trajectory = np.array([x_lst[t][cell] for t in range(len(x_lst))])
        axs[0].plot(trajectory[:,0], trajectory[:,1])

    # Plot trajectory with time color gradient:

    max_t = len(x_lst)

    for i in range(len(tracked_idx)):
        cell = tracked_idx[i]
        trajectory = np.array([x_lst[t][cell] for t in range(max_t)])

        # build line segments
        points = trajectory[:, :2].reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # color values along time
        colors = np.linspace(0, 1, max_t-1)

        lc = LineCollection(segments, cmap='viridis')
        lc.set_array(colors)

        axs[1].add_collection(lc)

    # Plot trajectory with z color gradient:


    for i in range(len(tracked_idx)):
        cell = tracked_idx[i]
        trajectory = np.array([x_lst[t][cell] for t in range(max_t)])
        z_vals = trajectory[:,2]

        # build line segments
        points = trajectory[:, :2].reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        segment_vals = z_vals[:-1]

        # color values along time
        colors = np.linspace(0, 1, max_t-1)

        lc = LineCollection(segments, cmap='mako')
        lc.set_array(segment_vals)

        axs[3].add_collection(lc)


    for i in range(len(tracked_idx)):
        cell = tracked_idx[i]
        trajectory = np.array([x_lst[t][cell] for t in range(len(x_lst))])
        mask_list = [p_mask_lst[t][cell] for t in range(len(x_lst))]
        axs[2].scatter(trajectory[:,0], trajectory[:,1],c=mask_list,s=1)



    plt.show()
    return 


### Plot avg. direction vector for each position ###
def flow_lines(file, resolution, plt_lim, if_df, start_frame=None, end_frame=None,frame_jump = 1, output_fil=None):
    '''
    Plotting flow lines for each position, averaged over cells and time:

    Inputs:
        file - path to file of sim results
        resolution: number of points along each axis
        plt_lim - limit for plot axes (limit for P-D embryo projection)
        if_df - whether or not to save df
        start_frame, end_frame - time range to average over (if None, averages over all time points)
        frame_jump - how many frames to jump for calculating flow (default 1, i.e. calculate flow between consecutive frames)
        output_fil - path to the saved file
    Returns:
        2D quiver plot of average direction vectors at each position
        csv file if if_df is True

    '''

    with open(file,'rb') as f:
        p_mask_lst, x_lst, p_lst, q_lst, U_lst = pickle.load(f)
    

    dx = (plt_lim[1]-plt_lim[0])/resolution
    dy = (plt_lim[3]-plt_lim[2])/resolution
    x_min = plt_lim[0]
    y_min = plt_lim[2]  
    x_max = plt_lim[1]
    y_max = plt_lim[3]
    range_x = np.arange(x_min,x_max+dx,dx)
    range_y = np.arange(y_min,y_max+dy,dy)

    #Define color bar:

    cmap = plt.get_cmap("mako")      # choose colormap

    if if_df:
        df = pd.DataFrame(columns=['position_x','position_y','avg_vector_x','avg_vector_y','altitude'])

    plt.figure(figsize=(8,8))
    Altitude = []
    for i in range(len(range_x)-1):
        for j in range(len(range_y)-1):
            box_x_min = range_x[i]
            box_x_max = range_x[i+1]
            box_y_min = range_y[j]
            box_y_max = range_y[j+1]

            delta_x = 0
            delta_y = 0
            altitude_sum = 0
            ct = 0
            for t in range(start_frame if start_frame is not None else 0, end_frame-frame_jump if end_frame is not None else len(x_lst)-frame_jump,frame_jump):
                ve_mask = p_mask_lst[t]!=0
                x_t = x_lst[t][ve_mask]
                x_t1 = x_lst[t+frame_jump][ve_mask]

                cells_in_box = np.where((x_t[:,0]>=box_x_min) & (x_t[:,0]<box_x_max) & (x_t[:,1]>=box_y_min) & (x_t[:,1]<box_y_max))[0]
                delta_x += np.sum(x_t1[cells_in_box,0] - x_t[cells_in_box,0])
                delta_y += np.sum(x_t1[cells_in_box,1] - x_t[cells_in_box,1])
                altitude_sum += np.mean(x_t[cells_in_box,2]) if len(cells_in_box)>0 else 0
                ct += 1 if len(cells_in_box)>0 else 0
            

            altitude = altitude_sum/ct if ct>0 else np.nan
            altitude = altitude/20 +1 # normalize altitude to [0,1] range assuming max altitude ~20
            
            norm = np.sqrt(delta_x**2 + delta_y**2)
            normalized_vector = (delta_x/norm, delta_y/norm) if norm !=0 else (0,0)
            if if_df:
                df = pd.concat([df, pd.DataFrame({'position_x':[ (box_x_min+box_x_max)/2 ],
                                                  'position_y':[ (box_y_min+box_y_max)/2 ],
                                                  'avg_vector_x':[ delta_x ],
                                                  'avg_vector_y':[ delta_y ],
                                                  'altitude':[ altitude ]})], ignore_index=True)
            alt_color = cmap(altitude)               # RGBA floats
            if not np.isnan(altitude):
                plt.quiver((box_x_min+box_x_max)/2, (box_y_min+box_y_max)/2, normalized_vector[0], normalized_vector[1], angles='xy', scale_units='xy', scale=1.5, color=matplotlib.colors.rgb2hex(alt_color), width=norm*0.001+0.0001)
            Altitude.append(altitude)
    df.to_csv(output_fil, index=False) if if_df else None
    return 
        



### plot gradient of selection ###

'''
Plotting a graident of Wnt at last time point in Voronoi looking down

Input:
    file: path to .npy file containing data on all time points
    cell_type (int from 0 to 2): cell type to include/exclude from plotta
    include(bool): whether to include (T) or exclude (F) the selected cell type
    VE_radius: radius for Voronoi plot bounds

Returns:
    plot of all morphogen graidents
'''

def plot_vor(file,cell_type,include,VE_radius):


    with open(file,'rb') as f:
        p_mask_lst, x_lst, p_lst, q_lst, U_lst = pickle.load(f)

    # Take last time point arrays
    p_mask = p_mask_lst[-1]
    x = x_lst[-1]
    U = U_lst[-1]

    # Create mask for cells of type 0 and select their positions
    if include:
        cell_mask = (p_mask == cell_type)
    else:
        cell_mask = (p_mask != cell_type)
    points = x[cell_mask]   # shape (n_cells, 3)

    if points.size == 0:
        raise ValueError("No cells with p_mask != 0 found at final time point.")
    # Project to 2D (x, y) for Voronoi (expects shape (n_points, 2))
    points_2D = points[:, :2]

    # Pick the scalar you want to color by (edit as needed)
    # If U is (N, k), choose e.g. channel 0:
    vals = U[cell_mask][:, 2]


    # ----------------------------
    # Disk boundary definition
    # ----------------------------
    center = np.array([0.0, 0.0])  # change if your disk isn't centered at origin
    radius = 1.02 * np.max(np.linalg.norm(points_2D - center, axis=1))  # auto radius

    disk = Point(center[0], center[1]).buffer(radius, resolution=256)

    # ----------------------------
    # Make Voronoi bounded: add ghost points on a big circle
    # ----------------------------
    n_ghost = 256
    ghost_scale = VE_radius  # bigger => safer bounding
    ang = np.linspace(0, 2*np.pi, n_ghost, endpoint=False)
    Rg = ghost_scale * radius
    ghost = np.c_[center[0] + Rg*np.cos(ang), center[1] + Rg*np.sin(ang)]

    all_pts = np.vstack([points_2D, ghost])
    vor = Voronoi(all_pts)

    # ----------------------------
    # Plot: color each clipped Voronoi region by vals
    # ----------------------------
    norm = colors.Normalize(vmin=np.min(vals), vmax=np.max(vals))
    cmap = cm.get_cmap("mako")

    fig, ax = plt.subplots()

    for i, v in enumerate(vals):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]

        # With ghost points this should be bounded, but keep a guard
        if -1 in region or len(region) == 0:
            continue

        poly = Polygon(vor.vertices[region]).intersection(disk)
        if poly.is_empty:
            continue

        xpoly, ypoly = poly.exterior.xy
        patch = MplPolygon(np.c_[xpoly, ypoly],
                        facecolor=cmap(norm(v)),
                        edgecolor="k",
                        linewidth=0.3)
        
        # Color type 2 cells gray instead of cmap
        if p_mask[cell_mask][i] == 2:
            patch = MplPolygon(np.c_[xpoly, ypoly],
                        facecolor="pink",
                        edgecolor="k",
                        linewidth=0.3)
        ax.add_patch(patch)

    # points + disk outline
    ax.scatter(points_2D[:, 0], points_2D[:, 1], s=8, c="k")
    ax.add_patch(plt.Circle((center[0], center[1]), radius, fill=False))

    # colorbar
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Value")

    ax.set_aspect("equal", "box")
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    plt.show()
    return



#Unwrapped gradients
def unwrap_grad(file,cell_type, include, timepoints, if_w):
    '''
    Plotting unwrapped gradients of all morphogens over time

    Input:
        file: path to .npy file containing data on all time points
        cell_type (int from 0 to 2): cell type to include/exclude from plotta
        include(bool): whether to include (T) or exclude (F) the selected cell type
        timepoints(list): list of timepoints to plot
        if_w: if we included w or not
    Returns:
        2D plot of all morphogen graidents over time
    '''

    if not if_w:
        with open(file, "rb") as f:
            p_mask_lst, x_lst, p_lst, q_lst, U_lst = pickle.load(f)
    else:
        with open(file, "rb") as f:
            p_mask_lst, x_lst, p_lst, q_lst, w_lst, U_lst = pickle.load(f)

    # First, let's find out the maximum concentration of all three morphogens throughout all time points

    dkk_max_I=[]
    wnt_max_I=[]
    for i in range(len(U_lst)):
        dkk_max_i = np.max(U_lst[i][:,0])
        dkk_max_I.append(dkk_max_i)
        wnt_max_i = np.max(U_lst[i][:,1])
        wnt_max_I.append(wnt_max_i)

    dkk_max = np.max(dkk_max_I)
    wnt_max = np.max(wnt_max_I)
        
    # Loop for different time points
    frames = timepoints

    # Generate subplots
    fig, axs = plt.subplots(len(frames),2,figsize=(12,12))

    ct = 0 # just a counter
    for i in frames:
        if not if_w:
            p_mask, x, p, q, U = p_mask_lst[i], x_lst[i], p_lst[i], q_lst[i], U_lst[i]
        else:
            p_mask, x, p, q, w, U = p_mask_lst[i], x_lst[i], p_lst[i], q_lst[i], w_lst[i], U_lst[i]
        #Slice the z-axis into slices with thickness of 1 starting from z_max to z_min
        z_min, z_max = np.min(x[:, 2]), np.max(x[:, 2])


        # For each slice, calculate the angle of the line connecting a cell (x,y,z) and its plane center (0,0,z) to the x-axis.
        # compute as a NumPy array so boolean masking works without TypeError
        angles = np.arctan2(x[:, 1], x[:, 0])  # Angle in radians

        if include:
            type_mask = (p_mask == cell_type)
        else:
            type_mask = (p_mask != cell_type) 
        x_type = x[type_mask]

        #Morphogens
        I_type = U[type_mask][:,0] #Inhibitor concentration
        W_type = U[type_mask][:,1]  # Wnt concentration 


        # generate plots

        axs[ct,0].scatter(angles[type_mask], x_type[:,2], c=I_type, cmap='viridis', s=20, vmin=0, vmax=dkk_max)

        axs[ct,1].scatter(angles[type_mask], x_type[:,2], c=W_type, cmap='plasma', s=20, vmin=0, vmax=wnt_max)
        if np.any(p_mask[type_mask]==3):
            axs[ct,1].scatter(angles[p_mask==3], x[p_mask==3][:,2],color="lightblue",s=20)


        ct = ct+1



    for i, ax in enumerate(axs.flat):
        if i<3*(len(frames)-1):
            ax.set_xticklabels([])
        else:
            ax.tick_params(axis='x',labelsize=7)

        if (i%3)!=0:
            ax.set_yticklabels([])
        else:
            ax.tick_params(axis='y',labelsize=7)


    axs[len(frames)-1,1].set_xlabel("Angles in radian",fontsize=10)
    axs[(len(frames)-1)//2,0].set_ylabel("z-slices",fontsize=10)



    # Loop over columns
    for j in range(2):
        # pick one scatter plot from this column (say, the first row)
        sc = axs[0, j].collections[0]   # first scatter in that Axes

        # make a colorbar for the whole column
        cbar = fig.colorbar(sc, ax=axs[:, j],
                            location="top", orientation="horizontal",
                            shrink=0.7, pad=0.01)

        cbar.ax.tick_params(labelsize=8)
        cbar.ax.xaxis.set_ticks_position("top")
        cbar.ax.xaxis.set_label_position("top")


    col_titles = ["DKK", "Wnt"]

    for j, title in enumerate(col_titles):
        axs[0, j].set_title(title, fontsize=12, pad=30)  # pad=30 moves it up a bit



    for i, val in enumerate(frames):
        axs[i, -1].annotate(
            f"t={val}",              # <-- use the value
            xy=(1.05, 0.5),
            xycoords="axes fraction",
            va="center", ha="left",
            fontsize=10
        )

#Unwrapped gradients
def unwrap_grad3(file,cell_type, include, timepoints, if_w):
    '''
    Plotting unwrapped gradients of all morphogens over time

    Input:
        file: path to .npy file containing data on all time points
        cell_type (int from 0 to 2): cell type to include/exclude from plotta
        include(bool): whether to include (T) or exclude (F) the selected cell type
        timepoints(list): list of timepoints to plot
        if_w: if we included w or not
    Returns:
        2D plot of all morphogen graidents over time
    '''

    if not if_w:
        with open(file, "rb") as f:
            p_mask_lst, x_lst, p_lst, q_lst, U_lst = pickle.load(f)
    else:
        with open(file, "rb") as f:
            p_mask_lst, x_lst, p_lst, q_lst, w_lst, U_lst = pickle.load(f)

    # First, let's find out the maximum concentration of all three morphogens throughout all time points
    bmp_max_I=[]
    dkk_max_I=[]
    wnt_max_I=[]
    for i in range(len(U_lst)):
        bmp_max_i = np.max(U_lst[i][:,0])
        bmp_max_I.append(bmp_max_i)
        dkk_max_i = np.max(U_lst[i][:,1])
        dkk_max_I.append(dkk_max_i)
        wnt_max_i = np.max(U_lst[i][:,2])
        wnt_max_I.append(wnt_max_i)

    bmp_max = np.max(bmp_max_I)
    dkk_max = np.max(dkk_max_I)
    wnt_max = np.max(wnt_max_I)
        
    # Loop for different time points
    frames = timepoints

    # Generate subplots
    fig, axs = plt.subplots(len(frames),3,figsize=(12,12))

    ct = 0 # just a counter
    for i in frames:
        if not if_w:
            p_mask, x, p, q, U = p_mask_lst[i], x_lst[i], p_lst[i], q_lst[i], U_lst[i]
        else:
            p_mask, x, p, q, w, U = p_mask_lst[i], x_lst[i], p_lst[i], q_lst[i], w_lst[i], U_lst[i]
        #Slice the z-axis into slices with thickness of 1 starting from z_max to z_min
        z_min, z_max = np.min(x[:, 2]), np.max(x[:, 2])


        # For each slice, calculate the angle of the line connecting a cell (x,y,z) and its plane center (0,0,z) to the x-axis.
        # compute as a NumPy array so boolean masking works without TypeError
        angles = np.arctan2(x[:, 1], x[:, 0])  # Angle in radians

        if include:
            type_mask = (p_mask == cell_type)
        else:
            type_mask = (p_mask != cell_type) 
        x_type = x[type_mask]

        #Morphogens]\
        B_type = U[type_mask][:,0]
        I_type = U[type_mask][:,1] #Inhibitor concentration
        W_type = U[type_mask][:,2]  # Wnt concentration 


        # generate plots

        axs[ct,1].scatter(angles[type_mask], x_type[:,2], c=I_type, cmap='viridis', s=20, vmin=0, vmax=dkk_max)

        axs[ct,2].scatter(angles[type_mask], x_type[:,2], c=W_type, cmap='plasma', s=20, vmin=0, vmax=wnt_max)
        axs[ct,0].scatter(angles[type_mask], x_type[:,2], c=B_type, cmap='cividis', s=20, vmin=0, vmax=bmp_max) 
        if np.any(p_mask[type_mask]==2):
            axs[ct,2].scatter(angles[p_mask==2], x[p_mask==2][:,2],color="lightblue",s=20)


        ct = ct+1



    for i, ax in enumerate(axs.flat):
        if i<3*(len(frames)-1):
            ax.set_xticklabels([])
        else:
            ax.tick_params(axis='x',labelsize=7)

        if (i%3)!=0:
            ax.set_yticklabels([])
        else:
            ax.tick_params(axis='y',labelsize=7)


    axs[len(frames)-1,1].set_xlabel("Angles in radian",fontsize=10)
    axs[(len(frames)-1)//2,0].set_ylabel("z-slices",fontsize=10)



    # Loop over columns
    for j in range(3):
        # pick one scatter plot from this column (say, the first row)
        sc = axs[0, j].collections[0]   # first scatter in that Axes

        # make a colorbar for the whole column
        cbar = fig.colorbar(sc, ax=axs[:, j],
                            location="top", orientation="horizontal",
                            shrink=0.7, pad=0.01)

        cbar.ax.tick_params(labelsize=8)
        cbar.ax.xaxis.set_ticks_position("top")
        cbar.ax.xaxis.set_label_position("top")


    col_titles = ["BMP",'DKK', "Wnt"]

    for j, title in enumerate(col_titles):
        axs[0, j].set_title(title, fontsize=12, pad=30)  # pad=30 moves it up a bit



    for i, val in enumerate(frames):
        axs[i, -1].annotate(
            f"t={val}",              # <-- use the value
            xy=(1.05, 0.5),
            xycoords="axes fraction",
            va="center", ha="left",
            fontsize=10
        )






# Napari animation
def animate(file, morph, DVE_type=3, if_w = True, no_morph = 2, cam_angle=(90,0,0), frame_t = 20):
    # load file
    if if_w:
        with open(file , 'rb') as f:
            p_mask_lst, x_lst, p_lst, q_lst, w_lst, U_lst = pickle.load(f)
    else: 
         with open(file , 'rb') as f:
            p_mask_lst, x_lst, p_lst, q_lst, U_lst = pickle.load(f)       

    # --- inputs you already have ---
    # x_lst: list/array length T, each (N,3)
    # U_lst: list/array length T, each (N,3) (or at least [:,2] exists)
    # p_mask_lst: list/array length T, each (N,) int mask
    # q_lst: list/array length T, each (N,3) vector for each point

    T = len(U_lst)
    viewer = napari.Viewer(ndisplay=3)
    viewer.camera.angles = cam_angle
    viewer.camera.zoom = 10
    viewer.theme = "light"
    anim = Animation(viewer)
    # global fixed color range across the whole animation
    vmin =0
    vmax = max(np.max(u[:, morph]) for u in U_lst)
    d_angle = 0   # degrees per frame (tune this if you want the camera to rotate consistently)
    base_angles = viewer.camera.angles  # starting angles, e.g. (90, 0, 0)

    # visual scale for vectors (tune this)
    vec_scale = 2



    # initialize frame 0
    t = 0
    x0 = x_lst[t]
    b0 = U_lst[t][:, morph]
    m0 = p_mask_lst[t]
    q0 = q_lst[t]
    p0 = p_lst[t]
    if no_morph == 2:
        morph_list = ['DKK','Wnt']
    if no_morph ==3:
        morph_list = ['BMP','DKK','Wnt']
    epi_cells = viewer.add_points(
        x0[m0==0],
        size=3,
        name="Epi",
        properties={morph_list[morph]: b0[m0==0]},
        face_color=morph_list[morph],
        face_colormap="viridis",
        face_contrast_limits=(vmin, vmax),
    )

    ve_cells = viewer.add_points(
        x0[m0!=0],
        size=3,
        name="VE",
        properties={morph_list[morph]: b0[m0!=0]},
        face_color=morph_list[morph],
        face_colormap="viridis",
        face_contrast_limits=(vmin, vmax),
    )
    if DVE_type==3:
        ExVe_cells = viewer.add_points(
            x0[m0==2],
            size=3,
            name="EXVE",
            face_color='black',
        )

    dve = viewer.add_points(
        x0[m0 == DVE_type],
        size=3,
        name="DVE",
        face_color="#F24464",
    )


    # vectors layer expects (M, 2, D): [ [start, end], ... ]
    vec_data0 = np.stack([x0[m0==DVE_type], -vec_scale * q0[m0==DVE_type]], axis=1)  # (N,2,3)
    vecs = viewer.add_vectors(
        vec_data0,
        name="q vectors",
        edge_width=1,
        edge_color="black",
    )

    vec_data1 = np.stack([x0[m0!=0], -vec_scale * p0[m0!=0]], axis=1)  # (N,2,3)
    vecs1 = viewer.add_vectors(
        vec_data1,
        name="ABP vectors",
        edge_width=1,
        edge_color="blue",
    )



    # --- animation timer ---
    state = {"t": 0, "playing": True}

    def set_frame(t):
        x = x_lst[t]
        b = U_lst[t][:, morph]
        m = p_mask_lst[t]
        q = q_lst[t]
        p = p_lst[t]

        epi_cells.data = x[m==0]
        epi_cells.properties = {morph_list[morph]: b[m==0]}
        # keep range fixed (sometimes useful to re-assert)
        epi_cells.face_contrast_limits = (vmin, vmax)

        ve_cells.data = x[m!=0]
        ve_cells.properties = {morph_list[morph]: b[m!=0]}
        # keep range fixed (sometimes useful to re-assert)
        ve_cells.face_contrast_limits = (vmin, vmax)
        if if_w:
            ExVe_cells.data = x[m==2]
        dve.data = x[m == DVE_type]

        vecs.data = np.stack([x[m==DVE_type], -vec_scale * q[m==DVE_type]], axis=1)
        vecs1.data = np.stack([x[m!=0], -vec_scale * p[m!=0]], axis=1)
        a0, a1, a2 = viewer.camera.angles
        viewer.camera.angles = (a0, a1, a2+d_angle)

        # optional: show frame index
        viewer.text_overlay.text = f"frame {t+1}/{T}"



    def advance():
        if not state["playing"]:
            return
        state["t"] = (state["t"] + 1) % T
        set_frame(state["t"])

    timer = QTimer()
    timer.setInterval(frame_t)  # ms per frame (20 fps). change to taste
    timer.timeout.connect(advance)
    timer.start()

    # handy controls
    @viewer.bind_key("p")
    def toggle_play(viewer):
        state["playing"] = not state["playing"]

    @viewer.bind_key("Right")
    def next_frame(viewer):
        state["playing"] = False
        state["t"] = (state["t"] + 1) % T
        set_frame(state["t"])

    @viewer.bind_key("d")
    def next_frame(viewer):
        state["playing"] = False
        state["t"] = (state["t"] + 20) % T
        set_frame(state["t"])

    @viewer.bind_key("Left")
    def prev_frame(viewer):
        state["playing"] = False
        state["t"] = (state["t"] - 1) % T
        set_frame(state["t"])

    @viewer.bind_key("a")
    def next_frame(viewer):
        state["playing"] = False
        state["t"] = (state["t"] - 20) % T
        set_frame(state["t"])

    napari.run()
