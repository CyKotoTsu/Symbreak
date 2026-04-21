 ### Imports ###
import numpy as np
import torch
from scipy.spatial import cKDTree
import os
import itertools
import pickle
from time import time
import json


### Simulation ###
class Simulation:
    """
    Simulation class for running a cell-based model with polarity-based interactions

    This class handles the initialization and execution of the simulation, including the management
    of cell states, interactions, and the application of forces based on cell polarity.
    """

    def __init__(self, sim_dict):
        """
        Initializes the simulation with the given parameters.

        Parameters:
            sim_dict (dict): A dictionary containing simulation parameters.

        Returns:
            None
        """

        # Stuff for tensors
        self.device         = sim_dict['device']            # Device for tensor operations. 'cuda' for GPU acceleration otherwise 'cpu'
        self.dtype          = sim_dict['dtype']             # Data type for tensors. float32 or float64

        # Simulation parameters
        self.k              = 100                           # Number of nearest neighbors to consider
        self.true_neighbour_max     = 50                    # Maximum number of true neighbors in former timestep
        self.dt            = sim_dict['dt']                # Size of timestep for simulation
        self.sqrt_dt        = np.sqrt(self.dt)              # Square root of time step. We calculate it here instead of in the update loop
        self.rate_diff      = sim_dict['rate_diff']          # How many timesteps between morphogen diffusion updates
        self.eta            = sim_dict['eta']               # Noise strength
        self.lambdas        = torch.tensor(sim_dict['lambdas'], device=self.device, dtype=self.dtype)   # Lambda values determining cell interactions
        self.max_cells      = sim_dict['max_cells']         # Maximum number of cells in the simulation. Once this is reached, the simulation terminates
        self.prolif_delay   = sim_dict['prolif_delay']      # Delay before cell proliferation
        self.yield_every    = sim_dict['yield_every']       # How many timesteps between data yields
        self.random_seed    = sim_dict['random_seed']       # Random seed for reproducibility
        self.prolif_rate    = sim_dict['prolif_rate']       # Probability of cell proliferation for each cell
        self.egg_shape      = sim_dict['egg_shape']      # The shape of the egg (for scaling positions if using boundary conditions)
        self.ceiling_z      = sim_dict['ceiling_z']         # Where the ceiling is positioned 
        self.z_wall_k       = sim_dict['z_wall_k']          # The ceiling spring constant
        self.push           = sim_dict['push']              # Active force for cellular motion intensity
        self.t_vstop        = sim_dict['t_vstop']          # Time to stop the DVE migration
        self.v_growth        = sim_dict['v_growth']           # Growth rate of the embryo axes (used for cell division)
        # stuff we need to initialize
        self.d      = None                                  # Distances to nearest neighbors
        self.idx    = None                                  # Indices of nearest neighbors
        self.beta   = None                                  # Tensor used for cell division
        self.init_number = sim_dict.get('init_number', 0)          # To keep track of which initial condition we are using

        #Stuff for diffusion
        self.c_bmp = sim_dict['c_bmp']                      # BMP diffusion constant
        self.c_dkk = sim_dict['c_dkk']                      # DKK diffusion constant
        self.c_wnt = sim_dict['c_wnt']                      # Wnt diffusion constant
        self.alpha_bmp = sim_dict['alpha_bmp']              # BMP source rate
        self.a_wnt = sim_dict['a_wnt']                      # Wnt production rate
        self.a_dkk = sim_dict['a_dkk']                      # DKK production rate in all cells       
        self.p_dkk = sim_dict['p_dkk']                    # DKK production in DVE cells
        self.hc = sim_dict['hc']                          # Hill coefficient for morphogen interactions
        self.d_crit = sim_dict['d_crit']                    # Distance cutoff for morphogen diffusion weights
        self.ceiling_threshold_bmp = sim_dict['ceiling_threshold_bmp']  # Distance from ceiling where BMP sources are located
        self.offsets        = torch.tensor(sim_dict['offsets'], device=self.device, dtype=self.dtype)           # Offsets for periodic boundary conditions

        # Stuff for DVE induction
        self.dve_spawn = sim_dict['dve_spawn']              # z-value below which DVE cells are spawned
        self.dve_die = sim_dict['dve_die']                # z-value above which DVE cells are killed

        # Set random seed
        torch.manual_seed(self.random_seed)                 # For reproducibility


    @staticmethod
    def find_potential_neighbours(x, k=100, distance_upper_bound=np.inf, workers=-1):
        """
        Finds k nearest neighbors for each point in x. This is done via cKDTree,
        which is constructed on CPU. This should at some point be rectified to
        take place on the GPU.

        Parameters:
            x (torch.Tensor): Input data points.
            k (int): Number of nearest neighbors to find.
            distance_upper_bound (float): Maximum distance for neighbors.
            workers (int): Number of parallel workers to use.

        Returns:
            d (torch.Tensor): Distances to the nearest neighbors not including the point itself.
                Dimensions: (len(x), k)
            idx (torch.Tensor): Indices of the nearest neighbors not including the point itself.
                Dimensions: (len(x), k)
        """

        tree = cKDTree(x)               # Constructing the KDTree
        d, idx = tree.query(x, k + 1, distance_upper_bound=distance_upper_bound, workers=workers)   # Querying the KDTree
        idx = np.clip(idx, 0, len(x) - 1)  # Ensure indices are within bounds
        return d[:, 1:], idx[:, 1:]     # Returning distances and indices of nearest neighbors

    def find_true_neighbours(self, d, dx):
        """
        Constructs a boolean voronoi mask to the neighbors found via find_potential_neighbours().

        Parameters:
            d (torch.Tensor): Distances to the nearest neighbors not including the point itself.
                Dimensions: (len(x), k)
            dx (torch.Tensor): Differences between points and their neighbors.
                Dimensions: (len(x), k, 3)

        Returns:
            voronoi_mask (torch.Tensor): Boolean mask indicating true neighbors.
                Dimensions: (len(x), k)
        """

        with torch.no_grad():           # Disable gradient tracking as we don't want the simulation to optimize this
            voronoi_masks = []          # We  need to batch the construction of voronoi masks to avoid memory issues. 
            i0 = 0                      # Start index for batching
            batch_size = 1024           # Batch size for constructing voronoi masks. Adjust according to available vRAM
            i1 = batch_size             # End index for batching
            while True:                 # Loop until all points are processed
                if i0 >= dx.shape[0]:   # If start index is beyond the last point, break
                    break

                n_dis = torch.sum((dx[i0:i1, :, None, :] / 2 - dx[i0:i1, None, :, :]) ** 2, dim=3)              # If any cell C is closer to the AB-midpoint than A or B, then A and B are not neighbors
                n_dis += 1000 * torch.eye(n_dis.shape[1], device=self.device, dtype=self.dtype)[None, :, :]     # Add a large value to the diagonal to avoid self-neighboring

                voronoi_mask = torch.sum(n_dis < (d[i0:i1, :, None] ** 2 / 4), dim=2) == 0                      # Construct voronoi mask
                voronoi_masks.append(voronoi_mask)                                                              # Append voronoi mask to the list

                if i1 > dx.shape[0]:
                    break
                i0 = i1
                i1 += batch_size
        voronoi_mask = torch.cat(voronoi_masks, dim=0)      # Concatenate all voronoi masks
        return voronoi_mask                                 # Return the final voronoi mask

    def get_neighbors(self,x):
        """
        Find the voronoi neighbors of all cells.
        And shorten the tensors to include only the minimum of false neighbors.

        Parameters:
            x (torch.Tensor): Input tensor of shape (N, 3) where N is the number of points.

        Returns:
            d (torch.Tensor): Distances to the nearest neighbors not including the point itself.
                Dimensions: (len(x), m), where m is the maximum number of true neighbors found.
            idx (torch.Tensor): Indices of the nearest neighbors not including the point itself.
                Dimensions: (len(x), m), where m is the maximum number of true neighbors found.
            voronoi_mask (torch.Tensor): Boolean mask indicating true neighbors.
                Dimensions: (len(x), m), where m is the maximum number of true neighbors found.
            dx (torch.Tensor): Differences between points and their neighbors.
                Dimensions: (len(x), m, 3)
    """

        idx = self.idx
        full_n_list = x[idx]                                      
        dx = x[:, None, :] - full_n_list
        voronoi_mask = self.find_true_neighbours(self.d, dx)

        # Minimize size of voronoi_mask and reorder idx and dx
        sort_idx = torch.argsort(voronoi_mask.int(), dim=1, descending=True)        # We sort the boolean voronoi mask in descending order, i.e 1,1,1,...,0,0
        voronoi_mask = torch.gather(voronoi_mask, 1, sort_idx)                      # Reorder voronoi_mask
        dx = torch.gather(dx, 1, sort_idx[:, :, None].expand(-1, -1, 3))            # Reorder dx
        idx = torch.gather(idx, 1, sort_idx)                                        # Reorder idx
        m = torch.max(torch.sum(voronoi_mask, dim=1)) + 1                           # Finding new maximum number of true neighbors
        self.true_neighbour_max = m                                                 # Saving it so we can use it again later
        voronoi_mask = voronoi_mask[:, :m]                                          # Shorten voronoi_mask
        dx = dx[:, :m]                                                              # Shorten dx
        idx = idx[:, :m]                                                            # Shorten idx

        # Normalise dx
        d = torch.sqrt(torch.sum(dx**2, dim=2))                                     # Calculate w. new ordering
        dx = dx / d[:, :, None]                                                     # Normalize dx (also new ordering)

        return d, dx, idx, voronoi_mask                                          # Return all the goods
    
    def rescale_s(self, S):
        """
        Rescales the S values to be between 0 and 1.
        Via the transformation S_rescaled = (S + 1) / 2
        The purpose is to decouple differential adhesion strengths from the relaxation lenghtscales.

        Parameters:
            S (torch.Tensor): Input S values.

        Returns:
            S_rescaled (torch.Tensor): Rescaled S values.
        """
        S_rescaled = (S + 1.0) / 2.0
        return S_rescaled

    def diffuse_morphogens(self, x, p_mask, c_bmp, c_dkk, c_wnt, 
                           alpha_bmp,a_wnt, a_dkk, p_dkk, hc, U):
        """
        Simulates the diffusion and decay of morphogens using a finite difference method.

        Parameters:
        x (torch.Tensor): The positions of the cells.
        p_mask (torch.Tensor or None): A mask indicating cell types. If None, all cells are treated the same.
        c_bmp (float): The diffusion constant for BMP.
        c_dkk (float): The diffusion constant for DKK.
        c_wnt (float): The diffusion constant for Wnt.
        alpha_bmp (float): The source rate for BMP.
        a_wnt (float): The production rate for Wnt.
        a_dkk (float): The production rate for DKK in all cells.
        p_dkk (float): The production rate for DKK in DVE cells.
        U (torch.Tensor): The current concentrations of the morphogens. Shape: (N, 3) where N is the number of cells and 3 corresponds to BMP, DKK, and Wnt.
        Returns:
        tuple: A tuple containing the updated concentrations of BMP, DKK, and Wnt morphogens.
        """

        #UNwieghted adjacency matrix based on cell distances
        ##Get the true neighbor distances, dx and more reordered and reduced
        d, dx, idx, voronoi_mask = self.get_neighbors(x)   # all torch tensors
        N, m = d.shape


        A = torch.zeros((N, N), dtype=self.dtype, device=self.device)


        for i in range(N):
            row_has_edge = False
            for k, j in enumerate(idx[i]):            # j: neighbor index
                if not voronoi_mask[i, k]:
                    continue
                if j == i or not torch.isfinite(d[i, k]):  # kill self-loops / bad distances
                    continue
                # Make the graph undirected 
                A[i, j] = 1 if d[i, k] < self.d_crit else 0
                A[j, i] = A[i, j]
                row_has_edge = True

        deg = A.sum(axis=1)

        D = deg
        D = torch.clamp(D, min=1e-12)  # Avoid division by zero by clamping degree to a small positive value
        L = torch.eye(N,  device=D.device, dtype=D.dtype) - (A / D[:, None])   # random-walk Laplacian


        # Source and sink matrix
        Theta = torch.zeros((len(x), 3), device=self.device, dtype=self.dtype) #Empty matrix for BMP, DKK and Wnt

        #In Theta matrix, select top cells near ceiling to be BMP sources
        ceiling_cells = torch.where((x[:,2] > (self.ceiling_z - self.ceiling_threshold_bmp)))[0]

        Theta[ceiling_cells, 2] += alpha_bmp  # BMP source

        #All morphogens decay
        Theta[:, 0] += -U[:, 0]  # BMP decay
        Theta[:, 1] += -U[:, 1]  # DKK decay
        Theta[:, 2] += -U[:, 2]  # Wnt decay


        LU = L @ U

        # # Replace NaN/Inf with finite values and clip extreme magnitudes to avoid overflow
        # LU = torch.nan_to_num(
        #     LU,
        #     nan=0.0,
        #     posinf=torch.finfo(LU.dtype).max / 1e6,
        #     neginf=torch.finfo(LU.dtype).min / 1e6,
        # )

        # Clip to a large but finite value to prevent later multiplications from overflowing
        max_allowed = 1e12
        if torch.any(torch.abs(LU) > max_allowed):
            LU = torch.clamp(LU, -max_allowed, max_allowed)

        #multiply each column with the respective diffusion constant
        diffusion = torch.zeros((len(x), 3), device=self.device, dtype=self.dtype)
        diffusion[:, 0] = -c_bmp * LU[:, 0]
        diffusion[:, 1] = -c_dkk * LU[:, 1]
        diffusion[:, 2] = -c_wnt * LU[:, 2]

        #compute dUdt = diffusion + Theta
        dUdt = diffusion + Theta

        #reaction term
        # wnt production is activated by BMP  and itself and repressed by DKK 
        dUdt[:,2] += a_wnt*(U[:,2]**hc/(1+U[:,2]**hc+U[:,1]**hc))
        # dkk production is activated by wnt via Hill function
        dUdt[:,1] += a_dkk*(U[:,2]**hc/(1+U[:,2]**hc+U[:,1]**hc))
        # DKK is constantly produced in DVE cells (type 2) and inhibits Wnt
        if p_mask is not None:
            type2_cells = torch.where(p_mask == 2)[0]
            dUdt[type2_cells, 1] += p_dkk  # DKK production in DVE cells

        return dUdt
    



    
    def egg_bound(self, pos, tstep):
        """
        Calculating additions to the potential due to spherical boundary conditions.
        Only does anything if self.bound_radius is a float.

        Parameters:
            pos (torch.Tensor): Input tensor of shape (N, 3) where N is the number of points.
            tstep (int): The current time step of the simulation

        Returns:
            V_add_sum (float): Addition to the potential energy due to spherical boundary conditions.

        """
        pos_vec = pos.clone() / torch.tensor(self.egg_shape)[None, :]  # Scaling the positions to the egg shape
        mag = torch.sum(pos_vec**2, dim=1)                     # Magnitude of the position vectors to the center of the egg
        v_add = torch.where(mag > (1.0+self.v_growth*tstep*self.dt)**2, (torch.exp(mag - (1.0+self.v_growth*tstep*self.dt)**2) - 1)*10., 0.) # Calculate additional potential energy

         # Z-wall: quadratic hinge (smooth, zero below, increasing above)
        z_plus = torch.where(pos[:, 2] > self.ceiling_z, pos[:, 2] - self.ceiling_z, torch.tensor(0.))            # max(z, 0)
        v_add = v_add + self.z_wall_k * (z_plus ** 2)          # choose z_wall_k (e.g., 100–1000)

        if torch.isnan(v_add).any() or torch.isinf(v_add).any():                        #check for nan or inf. This is mainly for debugging, but i've kept it as it sometimes does.... stuff. 
            print("Warning: NaN or Inf detected in potential energy")
        V_add_sum = v_add.sum()                                                         # Add it all up as we need it in scalar form for gradient computation.
        return V_add_sum

    def potential(self, x, p, q, p_mask, idx, tstep):
        """
        Calculate the potential energy between particles.

        Parameters:
            x (torch.Tensor): Positions of the particles.
            p (torch.Tensor): Apicobasal polarities of the cells
            q (torch.Tensor): Planar cell polarity of the cells
            p_mask (torch.Tensor): Mask denoting different cell types. Is bool in this implementation (only 2 cell types)
            idx (torch.Tensor): Indices of neighboring particles.
            tstep (int): The current time step of the simulation.

        Returns:
            V_sum (float): The total potential energy.
        """

        #Get the true neighbor distances, dx and more reordered and reduced
        d, dx, idx, voronoi_mask = self.get_neighbors(x)        

        if isinstance(p_mask, torch.Tensor):        # Check if p_mask is a tensor indicating 2 cell types. Otherwise it is None
            # Making interaction mask
            im_self = p_mask[:,None].expand(p_mask.shape[0], idx.shape[1])+1  # Making the interaction mask for selv-interaction, '+1' to make multiplication specific.
            im_nb = p_mask[idx]+1  # Same for interactions with neighbors
            interaction_mask = im_self*im_nb
            # Make lambda_self array
            l = torch.zeros(size=(interaction_mask.shape[0], interaction_mask.shape[1], 4),
                        device=self.device, dtype=self.dtype)   # Empty lambda tensor
            # Filling a lambda array with the right interactions

            l[interaction_mask == 1] = self.lambdas[0]          # type0-type0 interaction
            l[interaction_mask == 4] = self.lambdas[1]          # type1-type1 interaction
            l[interaction_mask == 9] = self.lambdas[2]          # type2-type2 interaction
            l[interaction_mask == 2] = self.lambdas[3]          # type0-type1 interaction
            l[interaction_mask == 3] = self.lambdas[4]          # type0-type2 interaction
            l[interaction_mask == 6] = self.lambdas[5]          # type1-type2 interaction

        else:
            # If multiple lambda array exist this is an error
            assert self.lambdas.ndim == 1, "Multiple lambda arrays found"
            l = self.lambdas[None, None, :].expand(x.shape[0], idx.shape[1], 4)  # Using the same lambda array for all particles

        # Expanding ABP and PCP for easy cross products
        pi = p[:, None, :].expand(p.shape[0], idx.shape[1], 3)
        pj = p[idx]
        qi = q[:, None, :].expand(q.shape[0], idx.shape[1], 3)
        qj = q[idx]

        # All the S-terms are calculated
        S1 = torch.sum(torch.cross(pj, dx, dim=2) * torch.cross(pi, dx, dim=2), dim=2)      # Calculating S1 (The ABP-position part of S). Scalar for each particle-interaction. Meaning we get array of size (n, m) , m being the max number of nearest neighbors for a particle
        S1 = self.rescale_s(S1)                                                             # Rescaling S1 to be between 0 and 1
        S2 = torch.sum(torch.cross(pi, qi, dim=2) * torch.cross(pj, qj, dim=2), dim=2)      # Calculating S2 (The ABP-PCP part of S).
        S2 = self.rescale_s(S2)                                                             # Rescaling S2 to be between 0 and 1
        S3 = torch.sum(torch.cross(qi, dx, dim=2) * torch.cross(qj, dx, dim=2), dim=2)      # Calculating S3 (The PCP-position part of S)
        S3 = self.rescale_s(S3)    

        # Calculating S
        S = l[:,:,0] + l[:,:,1] * S1 + l[:,:,2] * S2 + l[:,:,3] * S3

        if not torch.any(self.offsets):
            Vij = voronoi_mask.float() * S * (torch.exp(-d) -  torch.exp(-d/5))     # Calculating the potential energy between particles masking out false interactions via voronoi_mask
        else:
            offset_tensor = torch.zeros_like(d)                                     # Initializing offset tensor
            offset_tensor[interaction_mask == 1] = self.offsets[0]                  # Assigning offsets for 0-0
            offset_tensor[interaction_mask == 4] = self.offsets[1]                  # Assigning offsets for 1-1
            offset_tensor[interaction_mask == 9] = self.offsets[2]                  # Assigning offsets for 2-2
            offset_tensor[interaction_mask == 2] = self.offsets[3]                  # Assigning offsets for 0-1
            offset_tensor[interaction_mask == 3] = self.offsets[4]                  # Assigning offsets for 0-2
            offset_tensor[interaction_mask == 6] = self.offsets[5]                  # Assigning offsets for 1-2
            Vij = voronoi_mask.float() * S * (torch.exp(-d - offset_tensor) -  torch.exp((-d - offset_tensor)/5))     # Calculating the potential energy between particles masking out false interactions via voronoi_mask

        Vij_sum = torch.sum(Vij)   

        # Utilize spherical boundary conditions?
        if self.egg_shape:               # If self.egg_shape is set, we apply spherical boundary conditions
            bc = self.egg_bound(x, tstep)
        else:
            bc = 0.

        V = Vij_sum + bc

        return V

    def init_simulation(self, x, p, q, p_mask, U):
        """
        Initiating simulation parameters by transforming ndarrays to tensors and the like.

        Parameters:
            x (np.ndarray): Cell positions.
            p (np.ndarray): Apicobasal polarities.
            q (np.ndarray): Planar cell polarities
            p_mask (np.ndarray) or None: Mask denoting different cell types. If None, all cells are considered the same type.
            U (np.ndarray): Morphogen concentrations. (N,3) array where N is number of cells and 3 is for BMP, Nodal and Wnt.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The initialized tensors for cell positions, apicobasal polarities, planar cell polarities, and the particle mask.
            OR
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]: If p_mask is None, the last element is None.

        """

        # Check input lengths
        assert len(x) == len(p)
        assert len(q) == len(x)
        assert len(U) == len(x) # Morphogen concentrations must match number of cells

        # Putting the data on the right device (GPU or CPU)
        x = torch.tensor(x, requires_grad=True, dtype=self.dtype, device=self.device)
        p = torch.tensor(p, requires_grad=True, dtype=self.dtype, device=self.device)
        q = torch.tensor(q, requires_grad=True, dtype=self.dtype, device=self.device)
        U = torch.tensor(U, dtype=self.dtype, device=self.device)

        # We check if p_mask is np.ndarray or None
        if isinstance(p_mask, np.ndarray):
            p_mask = torch.tensor(p_mask, dtype=torch.int, device=self.device)
        else:
            p_mask = None       # Doubly making sure p_mask is None (not strictly necessary)
        self.beta   = torch.zeros(x.shape[0], dtype=self.dtype, device=self.device) # Initialization of beta tensor. Used for cell division.

        return x, p, q, p_mask, U  # Returning the goods.
    
    def update_k(self, true_neighbour_max):
        """
        Updates the number of k nearest neighbors found in find_potential_neighbours()
        based on the maximum number of true neighbors found in the last iteration.

        Parameters:
            true_neighbour_max (int): The maximum number of true neighbors found.

        Returns:
            k (int): The updated number of k nearest neighbors.

        """

        k = self.k
        fraction = true_neighbour_max / k       # Fraction between the maximimal number of nearest neighbors and the initial nunber of nearest neighbors we look for.
        if fraction < 0.25:                     # If fraction is small our k is too large and we make k smaller
            k = int(0.75 * k)
        elif fraction > 0.75:                   # Vice versa
            k = int(1.5 * k)
        self.k = k                              # We update k
        return k
    
    def update_neighbors_bool(self, tstep, division):
        """
        Returns whether to update potential neighbors or not

        Parameters:
            tstep (int): The current time step.
            division (bool): Whether cell division has occured in this timestep

        Returns:
            bool: Whether to update potential neighbors or not.

        """

        if division == True or tstep < 5_000:       # If cell division has occurred or we are in the early stages
            return True
        elif self.idx is None:                      # If we have not found any neighbors yet
            return True
        return (tstep % 20 == 0)                    # Otherwise we update every 20th step
    


    def time_step_morph(self,x, p_mask, U):
        """
        Advances the morphogen concentrations by one time step.

        Parameters:
            x (torch.Tensor): The cell positions.
            p_mask (torch.Tensor or None): The particle mask.
            U (torch.Tensor): The morphogen concentrations.

        Returns:
            U (torch.Tensor): The updated morphogen concentrations.
        """

        # Compute dUdt
        dUdt = self.diffuse_morphogens(
            x,
            p_mask,
            self.c_bmp,
            self.c_dkk,
            self.c_wnt,
            self.alpha_bmp,
            self.a_wnt,
            self.a_dkk,
            self.p_dkk,
            self.hc,
            U
        )

        # Update U using Euler's method
        U = U + (dUdt/self.rate_diff * self.dt)

        # Ensure non-negative concentrations
        U = np.maximum(U, 0.0)

        return U

    def time_step(self, x, p, q, p_mask, tstep, U):
        """
        Progresses the simulation by one time step.

        Parameters:
            x (torch.Tensor): The cell positions.
            p (torch.Tensor): The apicobasal polarities.
            q (torch.Tensor): The planar cell polarities.
            p_mask (torch.Tensor or None): The particle mask.
            tstep (int): The current time step.
            U (torch.Tensor): The morphogen concentrations.

        Returns:
            x (torch.Tensor): The updated cell positions.
            p (torch.Tensor): The updated apicobasal polarities.
            q (torch.Tensor): The updated planar cell polarities.
            p_mask (torch.Tensor or None): The updated particle mask.
            U (torch.Tensor): The updated morphogen concentrations.
        """
        # First induce and deduce DVE
        if self.dve_spawn is not None and self.dve_die is not None:
            prov_emve = torch.where((x[:,2]>self.dve_die) & (p_mask == 2))[0]
            p_mask[prov_emve] = 1
            prov_dve = torch.where((x[:,2]<self.dve_spawn) & (p_mask == 1))[0]
            p_mask[prov_dve] = 2

        # Update proliferation rate from when we want cell proliferation to occur
        if tstep == (self.prolif_delay + 1) and self.prolif_rate is not None:
            self.beta[:] = self.prolif_rate

        # Start with cell division
        division, x, p, q, p_mask, self.beta, U = self.cell_division(x, p, q, p_mask, U)

        # Idea: only update _potential_ neighbours every x steps late in simulation
        # For now we do this on CPU, so transfer will be expensive
        k = self.update_k(self.true_neighbour_max)      # Update k based on last iteration
        k = min(k, len(x) - 1)                          # No reason letting k be larger than number of cells
        if self.update_neighbors_bool(tstep, division): # Finding potential neighbors
            d,idx = self.find_potential_neighbours(x.detach().to("cpu").numpy(), k=k)
            self.idx = torch.tensor(idx, dtype=torch.long, device=self.device)
            self.d = torch.tensor(d, dtype=self.dtype, device=self.device)
        idx = self.idx

        # Calculate potential
        V = self.potential(x, p, q, p_mask, idx, tstep)

        # Backpropagation
        V.backward()


        # Cell positions and polarities are updated according to overdamped langevin dynamics. 
        with torch.no_grad():
            x += -x.grad * self.dt + self.eta * torch.empty(*x.shape, dtype=self.dtype, device=self.device).normal_() * self.sqrt_dt
            # For type2 cells add additional force
            if self.t_vstop is not None and tstep < self.t_vstop:
                x[p_mask==2] += -self.push*q[p_mask==2]*self.dt
            if self.t_vstop is None:
                x[p_mask==2] += -self.push*q[p_mask==2]*self.dt
            p += -p.grad * self.dt + self.eta * torch.empty(*p.shape, dtype=self.dtype, device=self.device).normal_() * self.sqrt_dt
            q += -q.grad * self.dt + self.eta * torch.empty(*q.shape, dtype=self.dtype, device=self.device).normal_() * self.sqrt_dt

        # We zero out the gradients for next time step
        p.grad.zero_()
        q.grad.zero_()
        x.grad.zero_()

        #normalize p and q after altering them in the update timestep
        eps = 1e-8
        with torch.no_grad():
            p /= torch.sqrt(torch.sum(p ** 2, dim=1))[:, None] + eps         # Normalizing p. Only the non-zero polarities are considered.
            q /= torch.sqrt(torch.sum(q ** 2, dim=1))[:, None] + eps         # Normalizing q. Only the non-zero polarities are considered.

        return x, p, q, p_mask, U  #Returning the goods.

    def simulation(self, x, p, q, p_mask, U):
        """
        Runs the simulation.

        Parameters:
            x (torch.Tensor): The cell positions.
            p (torch.Tensor): The apicobasal polarities.
            q (torch.Tensor): The planar cell polarities.
            p_mask (torch.Tensor or None): The particle mask.
            U (torch.Tensor): The morphogen concentrations.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor or None]: The updated cell positions, apicobasal polarities, planar cell polarities, and particle mask and morphogen concentrations.
        """
        
        #Initializing simulation
        x, p, q, p_mask, U = self.init_simulation(x, p, q, p_mask, U)

        tstep = 0
        while True:
            tstep += 1
            x, p, q, p_mask, U = self.time_step(x, p, q, p_mask, tstep, U)        #Advancing the simulation one timestep

            U = self.time_step_morph(x, p_mask, U)

            if tstep % self.yield_every == 0 or len(x) > self.max_cells:    #Yield data if we are at a 'yield step' or if we have too many cells and the simulation is aborted
                xx = x.detach().to("cpu").numpy().copy()                    #Copying data to CPU
                pp = p.detach().to("cpu").numpy().copy()
                qq = q.detach().to("cpu").numpy().copy()
                UU = U.detach().to("cpu").numpy().copy()
                if isinstance(p_mask, torch.Tensor):
                    pp_mask = p_mask.detach().to("cpu").numpy().copy()
                else:
                    pp_mask = None
                yield xx, pp, qq, pp_mask, UU                                  # Yielding the data 
    
    def cell_division(self, x, p, q, p_mask, U):
        """
        Handles cell division events.

        Parameters:
            x (torch.Tensor): The cell positions.
            p (torch.Tensor): The apicobasal polarities.
            q (torch.Tensor): The planar cell polarities.
            p_mask (torch.Tensor or None): The particle mask.
            U (torch.Tensor): The morphogen concentrations.

        Returns:
            Tuple[bool, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor or None, torch.Tensor]: A tuple containing:
                - division (bool): Whether division occurred.
                - x (torch.Tensor): Cell positions with new cells
                - p (torch.Tensor): Apicobasal polarities with new cells
                - q (torch.Tensor): Planar cell polarities with new cells
                - p_mask (torch.Tensor or None): The updated particle mask.
                - beta (torch.Tensor): The division probabilities.
                - U (torch.Tensor): The morphogen concentrations with new cells (unchanged in this implementation).
        """

        beta = self.beta            

        if torch.sum(beta) < 1e-8:              # If division probabilities are negligible no division occurs
            return False, x, p, q, p_mask, beta, U

        # set probability according to beta and dt
        d_prob = beta
        # flip coins
        draw = torch.empty_like(beta).uniform_()
        # find successes
        events = draw < d_prob
        division = False

        if torch.sum(events) > 0:
            with torch.no_grad():
                division = True
                # find cells that will divide
                idx = torch.nonzero(events)[:, 0]

                x0      = x[idx, :]
                p0      = p[idx, :]
                q0      = q[idx, :]
                U0      = U[idx, :]
                if isinstance(p_mask, torch.Tensor):
                    p_mask0 = p_mask[idx]
                b0      = beta[idx]

                # make a random vector
                move = torch.empty_like(x0).normal_()

                # place new cells
                x0 = x0 + move

                # append new cell data to the system state
                x = torch.cat((x, x0))
                p = torch.cat((p, p0))
                q = torch.cat((q, q0))
                U = torch.cat((U, U0))
                if isinstance(p_mask, torch.Tensor):
                    p_mask = torch.cat((p_mask, p_mask0))
                beta = torch.cat((beta, b0))

        x.requires_grad = True
        p.requires_grad = True
        q.requires_grad = True

        return division, x, p, q, p_mask, beta, U      #Returning the goods.
    

def save(data_tuple, name, output_folder):
    """
    Saves the simulation data to an .npy file.

    Parameters:
        data_tuple (Tuple): (p_mask, x, p, q, U)
        name (str): The name of the file (without extension).
        output_folder (str): The folder to save the file in.

    Returns:
        None, but saves the data
    """

    with open(f'{output_folder}/{name}.npy', 'wb') as f:
        pickle.dump(data_tuple, f)

def run_simulation(sim_dict):
    """
    External simulation runner.

    Parameters:
        sim_dict (dict): The simulation parameters and data.

    Returns:
        None
    """

    # Make the simulation runner object:
    data_tuple = sim_dict.pop('data')           # We don't want to save the data in a .json file so we pop it
    verbose    = sim_dict.pop('verbose')        # This is not really important info other, so we pop it too
    yield_steps, yield_every = sim_dict['yield_steps'], sim_dict['yield_every'] # How often to yield data

    assert len(data_tuple) == 5 or len(data_tuple) == 2, 'data must be tuple of either len 2 (for data generation) or 4 (for data input)'

    np.random.seed(sim_dict['random_seed'])     # Setting the random seed

    # The data tuple given in the simulation dictionary can either be pre-generated data
    # or it can be generated by a supplied data generation function
    if len(data_tuple) == 5:
        print('Using input data')
        p_mask, x, p, q, U = data_tuple            # Unpacking the data tuple
    else:
        # Data generation tuple construction: (data_gen, data_gen_args)
        print('Using data generation function')
        data_gen = data_tuple[0]                    
        p_mask, x, p, q, U = data_gen(*data_tuple[1])
        # if p_mask only has 1 type of entry we set it to None
        if isinstance(p_mask, np.ndarray) and np.unique(p_mask).size == 1:
            p_mask = None
        
    sim = Simulation(sim_dict)                  # Initializing an instance of the Simulation class
    runner = sim.simulation(x, p, q, p_mask, U)    # Making a runner

    output_folder = sim_dict['output_folder']   # The folder to save the output data

    # Create the output folder if it doesn't exist
    try: 
        os.mkdir(output_folder)
    except:
        pass

    # Initialize lists to store simulation data
    x_lst = [x]
    p_lst = [p]
    q_lst = [q]
    p_mask_lst = [p_mask]
    U_lst = [U]

    # Save the simulation dictionary
    with open(output_folder + '/sim_dict.json', 'w') as f:
        sim_dict['dtype'] = str(sim_dict['dtype'])
        json.dump(sim_dict, f, indent = 2)

    # Save the initial simulation data according to parameter values
    save((p_mask_lst, x_lst, p_lst,  q_lst, U_lst), name='data'+str(sim_dict['init_number']), output_folder=output_folder)

    # Print the notes if verbose
    if verbose:
        notes = sim_dict['notes']
        print('Starting simulation with notes:')
        print(notes)

    i = 0       # Initialize the iteration counter
    t1 = time() # We timing stuff

    # Saving data at intervals specified by yield_every:
    for xx, pp, qq, pp_mask, UU in itertools.islice(runner, yield_steps):
        i += 1
        if verbose:
            print(f'Running {i} of {yield_steps}   ({yield_every * i} of {yield_every * yield_steps})   ({len(xx)} cells)', end='\r')

        x_lst.append(xx)
        p_lst.append(pp)
        q_lst.append(qq)
        p_mask_lst.append(pp_mask)
        U_lst.append(UU)
        
        if len(xx) > sim_dict['max_cells']:
            break
        
        # Every 50 yield steps we dump the data
        if i % 50 == 0:
            save((p_mask_lst, x_lst, p_lst,  q_lst, U_lst), name='data'+str(sim_dict['init_number']), output_folder=output_folder)
    
    if verbose:
        print(f'Simulation done, saved {i} datapoints')
        print('Took', time() - t1, 'seconds')

    save((p_mask_lst, x_lst, p_lst,  q_lst, U_lst), name='data'+str(sim_dict['init_number']), output_folder=output_folder)  # Last iteration is saved

def make_random_sphere(N, type0_frac , radius=35):
    """
    Generates cells uniformly distributed within a sphere with randomly
    initialized polarities

    Parameters
        N (int): The number of cells to generate.
        type0_frac (float): The fraction of cells of type 0.
        radius (float): The radius of the sphere.

    Returns
        tuple: A tuple containing the following elements:
            - mask (np.ndarray): The mask indicating the type of each cell.
            - x (np.ndarray): The positions of the cells.
            - p (np.ndarray): The apicobasal polarities of the cells.
            - q (np.ndarray): The planar cell polarities of the cells
    """
    # Generate random positions within a sphere
    x = np.random.randn(N, 3)
    r = radius * np.random.rand(N)**(1/3.)
    x /= np.sqrt(np.sum(x**2, axis=1))[:, None]
    x *= r[:, None]

    # Generate radial AB polarities and random PCP
    cm = np.mean(x, axis=0) #Center of mass
    p0 = x - cm # Initial apicobasal polarity pointing outward from center of mass
    p = p0 / np.sqrt(np.sum(p0**2, axis=1))[:,None]  # Normalize (unit vectors)

    q = np.random.randn(N, 3)
    q /= np.sqrt(np.sum(q**2, axis=1))[:,None]

    # Generate random cell types with specified fractions
    mask = np.random.choice([0,1], p=[type0_frac, 1-type0_frac], size=N)                #Mask detailing which cells are which type

    sphere_data = (mask, x, p, q)
    return sphere_data      # Returning the goods.


def make_random_vesicles(N, type0_frac,r0=20, r1=35):
    # Generates cells uniformly distributed within a spherical shell
    # small vesicle with radius r0
    x0 = np.random.randn(int(N*type0_frac), 3)  
    x0/= np.sqrt(np.sum(x0**2, axis=1))[:, None] # Normalize to unit sphere
    x0*=r0 # Scale to inner radius

    # large vesicle with radius r1
    x1 = np.random.randn(int(N*(1-type0_frac)), 3)   
    x1/= np.sqrt(np.sum(x1**2, axis=1))[:, None] # Normalize to unit sphere
    x1*=r1 # Scale to outer radius

    #combine x0 and x1
    x = np.vstack((x0,x1))

    #assign cell types accordingly
    mask = np.array([0]*len(x0) + [1]*len(x1))

    # Generate radial AB polarities and random PCP
    cm = np.mean(x, axis=0) #Center of mass
    p0 = x - cm # Initial apicobasal polarity pointing outward from center of mass
    p = p0 / np.sqrt(np.sum(p0**2, axis=1))[:,None]  # Normalize (unit vectors)

    q = np.random.randn(N, 3)
    q /= np.sqrt(np.sum(q**2, axis=1))[:,None]

    vesicle_data = (mask, x, p, q)
    return vesicle_data      # Returning the goods.






