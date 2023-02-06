## Module for alpha shaping 

## Many functions originally written by Remi Proville https://bitbucket.org/benglitz/fishualizer_public/src/master/

import numpy as np
from numpy.linalg import norm, det
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
from itertools import combinations

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
# from mpl_toolkits.mplot3d.axes3d import Axes3D
from ipywidgets import interact

import sys 
import land_cover_analysis as lca
import land_cover_visualisation as lcv
import geopandas as gpd

def circum(points):
    """
    Compute the radius of the circum circle or sphere to the 3 or 4 given points
    """
    if points.shape[1]==2:
        n = 4
    else:
        n = 5
    M = np.ones((n, n))
    M[1:, :-1] = [[norm(p)**2, *p] for p in points]
    M11 = compute_minor(M, 0, 0)
    if M11 == 0:
        return np.inf
    
    M12 = compute_minor(M, 0, 1)
    M13 = compute_minor(M, 0, 2)
    M14 = compute_minor(M, 0, 3)
    x0 = 0.5 * M12 / M11
    y0 = - 0.5 * M13 / M11
    if n == 4:
        center = np.hstack((x0, y0))
    else:
        z0 = 0.5 * M14 / M11 
        center = np.hstack((x0, y0, z0))
    r = norm(points - center, axis=1)
    return r.mean(), center

def compute_minor(arr, i, j):
    """
    Compute minor of a matrix
    """
    assert type(arr) == np.ndarray and arr.ndim == 2
    rows = set(range(arr.shape[0]))
    rows.remove(i)
    cols = set(range(arr.shape[1]))
    cols.remove(j)
    sub = arr[np.array(list(rows))[:, np.newaxis], np.array(list(cols))]
    return det(sub)

def get_alpha_complex(simplices, points, alpha=.1, radii=None):
    if radii is None:
        radii = list(map(lambda s: circum(points[s])[0], simplices ))
        # get_alpha_complex.counter += 1
        # print(f"Radii computed {get_alpha_complex.counter} time(s)")
    return radii, [ix for ix, r in enumerate(radii) if r < alpha]
# get_alpha_complex.counter = 0
    
def random_complex(dt, points, alpha=.1):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_aspect(1)
    try:
        radii = random_complex._radii
    except AttributeError:
        radii = None
    radii, spx_ix = get_alpha_complex(dt.simplices, points, alpha=alpha, radii=radii)
    random_complex._radii = radii
    ax.scatter(*points.T);
    ax.triplot(points[:, 0], points[:, 1], dt.simplices[spx_ix])
    [ax.plot(*np.vstack((dt.points[p[0],:], dt.points[p[1],:])).T, c='k', linestyle='dotted') for p in dt.convex_hull]

def vertex_to_simplices(vertices, dt):
    '''### Getting all the simplices to which a vertex belongs'''
    simplices = {v: [] for v in vertices}
    for v in vertices:
        spx = dt.vertex_to_simplex[v]
        simplices[v].append(spx)
        to_explore = [x for x in dt.neighbors[spx] if x != -1]
        ix = 0
        while ix < len(to_explore):
            n = to_explore[ix]
            ix += 1
            if v in dt.simplices[n]:
                simplices[v].append(n)
                to_explore.extend([x for x in dt.neighbors[n] if x != -1 and x not in to_explore])
    return simplices

def circles_from_p1p2r(p1, p2, r):
    """
    Code from here: https://rosettacode.org/wiki/Circles_of_given_radius_through_two_points#Python
    Following explanation at http://mathforum.org/library/drmath/view/53027.html
    """
    if r == 0.0:
        raise ValueError('radius of zero')
    (x1, y1), (x2, y2) = p1, p2
    if all(p1 == p2):
        raise ValueError('coincident points gives infinite number of Circles')
    # delta x, delta y between points
    dx, dy = x2 - x1, y2 - y1
    # halfway point
    x3, y3 = (x1+x2)/2, (y1+y2)/2
    # dist between points
    q = np.sqrt(dx**2 + dy**2)
    if q > 2.0*r:
        # raise ValueError('separation of points > diameter')
        return (x3, y3), (x3, y3)
    # distance along the mirror line
    d = np.sqrt(r**2-(q/2)**2)
    # One answer
    c1 = (x3 - d*dy/q, y3 + d*dx/q)
    # The other answer
    c2 = (x3 + d*dy/q, y3 - d*dx/q)
    return c1, c2

def alpha_exposed_segments(simplex, dt, alpha):
    indices, indptr = dt.vertex_neighbor_vertices
    neigh = set(np.hstack([indptr[indices[p]:indices[p+1]] for p in simplex]))
    segments = []
    for pair in combinations(simplex, 2):
        c_neigh = neigh - set(pair)
        neigh_coords = dt.points[list(c_neigh),: ]
        centers = circles_from_p1p2r(dt.points[pair[0]], dt.points[pair[1]], alpha)
        dists = [cdist(np.atleast_2d(c), neigh_coords) for c in centers]
        exposed = [np.all(d > alpha) for d in dists]
        if exposed[0] ^ exposed[1]:
            segments.append(pair)
    return segments
    
def get_alpha_shape(spx_ix, dt, alpha):
    vert_in_ch = set(dt.convex_hull.reshape(-1))
    spx_in_cpx = set(spx_ix)
    vert_in_cpx = set(dt.simplices[spx_ix].reshape(-1))
    v_to_s = vertex_to_simplices(vert_in_cpx, dt)
    vert_in_shape = set()
    for v in vert_in_cpx: 
        if v in vert_in_ch:
            vert_in_shape.add(v)
            continue
        if all(s in spx_in_cpx for s in v_to_s[v]):
            continue
        vert_in_shape.add(v)
    # spx_in_shape = [list(filter(lambda v: v in vert_in_shape, dt.simplices[s])) for s in spx_in_cpx]
    spx_in_shape = set(sum([v_to_s[v] for v in vert_in_shape], []))
    segments = [alpha_exposed_segments(dt.simplices[spx_ix], dt, alpha) for spx_ix in spx_in_shape]
    segments = list(set(sum(segments, [])))
    return vert_in_shape, spx_in_shape, segments
        
def random_alpha_shape(dt, points, alpha=.13, ax=None, 
                       plot_hull=True, plot_inner_lines=True, plot_hull_points=True,
                       plot_convex_hull=True, plot_all_lines=True, plot_all_points=True):
    try:
        radii = random_alpha_shape._radii
    except AttributeError:
        radii = None
    # radii = None
    radii, spx_ix = get_alpha_complex(dt.simplices, points, alpha=alpha, radii=radii)
    random_alpha_shape._radii = radii
    vert_shape, spx_in_shape, seg_shape = get_alpha_shape(spx_ix, dt, alpha=alpha)
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
    ax.set_aspect(1)
    if plot_inner_lines:
        ax.triplot(points[:, 0], points[:, 1], dt.simplices[spx_ix])
    if plot_all_lines:
        ax.triplot(points[:, 0], points[:, 1], dt.simplices, linestyle='dotted', color=(.5,  .5, .5))
    if plot_hull:
        it = 0
        dict_tmp = {}
        for seg in seg_shape:
            pts = np.vstack([dt.points[s, :] for s in seg])
            dict_tmp[it] = pts.T
            ax.plot(*pts.T, c='k')
            it += 1
    if plot_all_points:
        ax.scatter(*dt.points.T, c='m', s=5, alpha=0.5)
    if plot_hull_points:
        ax.scatter(*dt.points[list(vert_shape), :].T, c='orange', s=70)
    if plot_convex_hull:
        [ax.plot(*np.vstack((dt.points[p[0],:], dt.points[p[1],:])).T, c='k', linestyle='dotted') for p in dt.convex_hull]
    # return ax, dict_tmp