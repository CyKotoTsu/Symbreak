
import pickle
import numpy as np
from scipy.spatial import cKDTree
import torch
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Set

# export neighbors:
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


def find_true_neighbours(d, dx):
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
            n_dis += 1000 * torch.eye(n_dis.shape[1], device='cpu', dtype=torch.float32)[None, :, :]     # Add a large value to the diagonal to avoid self-neighboring

            voronoi_mask = torch.sum(n_dis < (d[i0:i1, :, None] ** 2 / 4), dim=2) == 0                      # Construct voronoi mask
            voronoi_masks.append(voronoi_mask)                                                              # Append voronoi mask to the list

            if i1 > dx.shape[0]:
                break
            i0 = i1
            i1 += batch_size
    voronoi_mask = torch.cat(voronoi_masks, dim=0)      # Concatenate all voronoi masks
    return voronoi_mask                                 # Return the final voronoi mask

def get_neighbors(x, d0, idx0):
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


    full_n_list = x[idx0]                                      
    dx = x[:, None, :] - full_n_list
    voronoi_mask = find_true_neighbours(d0, dx)

    # Minimize size of voronoi_mask and reorder idx and dx
    sort_idx = torch.argsort(voronoi_mask.int(), dim=1, descending=True)        # We sort the boolean voronoi mask in descending order, i.e 1,1,1,...,0,0
    voronoi_mask = torch.gather(voronoi_mask, 1, sort_idx)                      # Reorder voronoi_mask
    dx = torch.gather(dx, 1, sort_idx[:, :, None].expand(-1, -1, 3))            # Reorder dx
    idx = torch.gather(idx0, 1, sort_idx)                                        # Reorder idx
    m = torch.max(torch.sum(voronoi_mask, dim=1)) + 1                           # Finding new maximum number of true neighbors
    voronoi_mask = voronoi_mask[:, :m]                                          # Shorten voronoi_mask
    dx = dx[:, :m]                                                              # Shorten dx
    idx = idx[:, :m]                                                            # Shorten idx

    # Normalise dx
    d = torch.sqrt(torch.sum(dx**2, dim=2))                                     # Calculate w. new ordering
    dx = dx / d[:, :, None]                                                     # Normalize dx (also new ordering)

    return d, dx, idx, voronoi_mask                                          # Return all the goods

# get idx and v_mask for all

def nb_to_arrays(x_lst):
    idxs = []
    v_masks = []
    for i in range(len(x_lst)):
        d0,idx0 = find_potential_neighbours(x_lst[i], k=100, distance_upper_bound=np.inf, workers=-1)
        d,dx,idx,v_mask = get_neighbors(torch.tensor(x_lst[i]), torch.tensor(d0), torch.tensor(idx0))
        idxs.append(idx)
        v_masks.append(v_mask)
    return idxs, v_masks


# ----------------------------
# 1) Build raw undirected edges from (T,N,M) neighbor lists + mask
# ----------------------------


Edge = Tuple[int, int]

def build_edge_sets(nbr_idx_list, v_mask_list) -> List[Set[Edge]]:
    """
    nbr_idx_list: List[torch.Tensor] where each is (N,M) int neighbor indices
    v_mask_list : List[torch.Tensor] where each is (N,M) bool mask

    Works with variable shapes per frame (different N and M across frames).

    Returns:
      E_raw[t] = set of undirected edges (i,j) with i<j present at frame t
    """
    if len(nbr_idx_list) != len(v_mask_list):
        raise ValueError(f"Number of frames mismatch: {len(nbr_idx_list)} vs {len(v_mask_list)}")

    E_raw = []

    for t in range(len(nbr_idx_list)):
        nbr_idx_t = nbr_idx_list[t]  # (N, M)
        v_mask_t = v_mask_list[t]    # (N, M)
        
        if nbr_idx_t.shape != v_mask_t.shape:
            raise ValueError(f"Frame {t}: nbr_idx shape {nbr_idx_t.shape} must match v_mask shape {v_mask_t.shape}")
        
        N = nbr_idx_t.shape[0]
        edges = set()
        
        for i in range(N):
            js = nbr_idx_t[i, v_mask_t[i]]  # neighbors for cell i at time t
            for j in js:
                j = int(j)
                if j < 0 or j >= N or j == i:
                    continue
                a, b = (i, j) if i < j else (j, i)
                edges.add((a, b))
        E_raw.append(edges)

    return E_raw

# ----------------------------
# 2) Debounce/hysteresis on edges to kill 1-frame Voronoi flicker
# ----------------------------
def debounce_edges(
    E_raw: List[Set[Edge]],
    K_on: int = 2,
    K_off: int = 2
) -> Tuple[List[List[Edge]], List[List[Edge]], List[Set[Edge]]]:
    """
    Hysteresis rules:
      - edge becomes stably ON after it is present for K_on consecutive frames
      - edge becomes stably OFF after it is absent  for K_off consecutive frames

    Returns:
      gained[t] = list of edges that turned stably ON at frame t
      lost[t]   = list of edges that turned stably OFF at frame t
      E_stable[t] = set of edges stably ON at frame t
    """
    if K_on < 1 or K_off < 1:
        raise ValueError("K_on and K_off must be >= 1")

    T = len(E_raw)
    gained = [[] for _ in range(T)]
    lost   = [[] for _ in range(T)]
    E_stable = [set() for _ in range(T)]

    stable: Dict[Edge, int] = defaultdict(int)       # stable state 0/1
    run_on: Dict[Edge, int] = defaultdict(int)       # consecutive present
    run_off: Dict[Edge, int] = defaultdict(int)      # consecutive absent

    all_edges = set().union(*E_raw) if T > 0 else set()

    for t in range(T):
        Et = E_raw[t]

        # update all edges that ever appear + edges currently stable ON
        edges_to_update = all_edges | {e for e, s in stable.items() if s == 1}

        for e in edges_to_update:
            present = (e in Et)

            if present:
                run_on[e] += 1
                run_off[e] = 0
            else:
                run_off[e] += 1
                run_on[e] = 0

            # transitions
            if stable[e] == 0 and run_on[e] >= K_on:
                stable[e] = 1
                gained[t].append(e)
            elif stable[e] == 1 and run_off[e] >= K_off:
                stable[e] = 0
                lost[t].append(e)

        # reconstruct stable edge set at time t
        E_stable[t] = {e for e, s in stable.items() if s == 1}

    return gained, lost, E_stable


# ----------------------------
# 3) Utilities: adjacency lookups from edge sets
# ----------------------------
def neighbors_from_edges(E: Set[Edge], N: int) -> List[Set[int]]:
    """
    Build adjacency list: adj[i] = set of neighbors of i from undirected edge set E
    """
    adj = [set() for _ in range(N)]
    for a, b in E:
        adj[a].add(b)
        adj[b].add(a)
    return adj


def edge_in(E: Set[Edge], a: int, b: int) -> bool:
    if a == b:
        return False
    x, y = (a, b) if a < b else (b, a)
    return (x, y) in E


# ----------------------------
# 4) Pair lost edges with gained edges into single T1-like intercalation events
# ----------------------------
def detect_T1_events(
    gained: List[List[Edge]],
    lost: List[List[Edge]],
    E_stable: List[Set[Edge]],
    N: int,
    match_window: int = 1,
    min_score: int = 2,
    p_mask_lst: List[np.ndarray] = None,
    direct_ant_indices: List[int] = None,
    direct_post_indices: List[int] = None
) -> List[Dict[str, Any]]:
    """
    Each T1-like event is detected by pairing:
      lost edge (a,b) at time t
      with gained edge (c,d) at time t' in [t, t+match_window]

    Locality / plausibility is enforced by a score computed from adjacency:
      - c and d should lie in the local neighborhood of a or b
      - and the 4-cell connectivity should look like a quadrilateral swap

    Parameters
    ----------
    match_window: allow gained edge to occur at t or up to t+match_window
                  (important if loss and gain do not happen exactly same frame)
    min_score: minimal connectivity score to accept a match (2 is a good default)

    Returns
    -------
    events: list of dicts, each describing one intercalation event
        {
          't_lost': t,
          't_gain': t',
          'lost_edge': (a,b),
          'gained_edge': (c,d),
          'cells': (a,b,c,d),
          'score': score
        }
    """
    T = len(E_stable)
    events = []

    # Precompute adjacency for each frame (stable graph)
    adj_list = [neighbors_from_edges(E_stable[t], N) for t in range(T)]

    for t in range(T):
        if not lost[t]:
            continue

        # candidate gained edges from t..t+match_window
        tg_max = min(T - 1, t + match_window)
        gained_candidates = []
        for tg in range(t, tg_max + 1):
            for e in gained[tg]:
                gained_candidates.append((tg, e))

        # keep track of which gained edges have been used already at each tg
        used_gained = set()  # entries are (tg, edge)

        # try to match each lost edge to one gained edge
        for (a, b) in lost[t]:
            # local neighborhood around a,b from stable graph around the transition
            # using both t and min(t+1,T-1) helps handle timing
            t2 = min(t + 1, T - 1)
            local = (adj_list[t][a] | adj_list[t][b] | adj_list[t2][a] | adj_list[t2][b] | {a, b})

            best = None
            best_score = -1

            for (tg, (c, d)) in gained_candidates:
                if (tg, (c, d)) in used_gained:
                    continue

                # locality filter: gained edge endpoints should be in local neighborhood
                if c not in local or d not in local:
                    continue

                # Score the "quadrilateral swap" plausibility.
                # We look at whether c and d connect to a and b around the transition.
                # Use stable graph at t2 (post-ish) as a proxy.
                Eref = E_stable[t2]

                score = 0
                score += 1 if edge_in(Eref, a, c) else 0
                score += 1 if edge_in(Eref, a, d) else 0
                score += 1 if edge_in(Eref, b, c) else 0
                score += 1 if edge_in(Eref, b, d) else 0

                # You can also add a constraint: gained endpoints should not equal lost endpoints
                if len({a, b, c, d}) < 4:
                    continue

                if score > best_score:
                    best_score = score
                    best = (tg, (c, d), score)

            if best is not None and best_score >= min_score:
                tg, gained_edge, score = best
                used_gained.add((tg, gained_edge))

                events.append({
                    "t_lost": t,
                    "t_gain": tg,
                    "lost_edge": (a, b),
                    "gained_edge": gained_edge,
                    "cells": (a, b, gained_edge[0], gained_edge[1]),
                    "score": score
                })

                # if ALL of (a, b, gained_edge[0], gained_edge[1]) is included in DVE cells, count as DVE intercalation
                dve_cells = set(np.where(p_mask_lst[0] == 2)[0])
                if all(c in dve_cells for c in (a, b, gained_edge[0], gained_edge[1])):
                    events[-1]["is_dve"] = True
                else:
                    events[-1]["is_dve"] = False

                # if ALL of (a, b, gained_edge[0], gained_edge[1]) is included in anterior cells, count as anterior intercalation
                ant_cells = set(direct_ant_indices)
                if all(c in ant_cells for c in (a, b, gained_edge[0], gained_edge[1])):
                    events[-1]["is_ant"] = True
                else:
                    events[-1]["is_ant"] = False

                # if ALL of (a, b, gained_edge[0], gained_edge[1]) is included in posterior cells, count as posterior intercalation
                post_cells = set(direct_post_indices)
                if all(c in post_cells for c in (a, b, gained_edge[0], gained_edge[1])):
                    events[-1]["is_post"] = True
                else:
                    events[-1]["is_post"] = False

      

    return events

# ----------------------------
# 5) Convenient wrapper + counting outputs
# ----------------------------
def analyze_intercalations(
    nbr_idx_list,
    v_mask_list,
    K_on: int = 2,
    K_off: int = 2,
    match_window: int = 1,
    min_score: int = 2,
    p_mask_lst: List[np.ndarray] = None,
    direct_ant_indices: List[int] = None,
    direct_post_indices: List[int] = None
) -> Dict[str, Any]:
    """
    Full pipeline:
      - raw edges
      - debounced edges -> stable edge sets
      - per-frame edge flip counts
      - T1-like event detection (pair lost/gained)

    Works with lists of torch tensors (variable shapes per frame).

    Returns dict with:
      E_raw, E_stable, gained, lost, edge_flip_counts, events, event_counts_per_frame
    """
    T = len(nbr_idx_list)
    N = nbr_idx_list[0].shape[0]  # cells in first frame

    E_raw = build_edge_sets(nbr_idx_list, v_mask_list)
    gained, lost, E_stable = debounce_edges(E_raw, K_on=K_on, K_off=K_off)

    edge_flip_counts = np.array([len(gained[t]) + len(lost[t]) for t in range(T)], dtype=int)

    events = detect_T1_events(
        gained=gained,
        lost=lost,
        E_stable=E_stable,
        N=N,
        match_window=match_window,
        min_score=min_score,
            p_mask_lst=p_mask_lst,
            direct_ant_indices=direct_ant_indices,
            direct_post_indices=direct_post_indices
    )

    event_counts_per_frame = np.zeros(T, dtype=int)
    for ev in events:
        event_counts_per_frame[ev["t_lost"]] += 1

    return {
        "E_raw": E_raw,
        "E_stable": E_stable,
        "gained": gained,
        "lost": lost,
        "edge_flip_counts": edge_flip_counts,          # counts edge activity
        "events": events,                              # discrete intercalation events
        "event_counts_per_frame": event_counts_per_frame
    }