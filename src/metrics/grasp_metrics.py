# !/usr/bin/env python -W ignore::DeprecationWarning
"""
Grasp Metrics for C106B Grasp Planning Lab
Author: Chris Correa
"""
import numpy as np
from utils import vec, adj, look_at_general, find_grasp_vertices, normal_at_point, find_intersections
from casadi import Opti, sin, cos, tan, vertcat, mtimes, sumsqr, sum1

# Can edit to make grasp point selection more/less restrictive
MAX_GRIPPER_DIST = .075
MIN_GRIPPER_DIST = .03

def compute_force_closure(vertices, normals, num_facets, mu, gamma, object_mass, mesh):
    """
    Compute the force closure of some object at contacts, with normal vectors 
    stored in normals. You can use the line method described in the project document.
    If you do, you will not need num_facets. This is the most basic (and probably least useful)
    grasp metric.

    Parameters
    ----------
    vertices : 2x3 :obj:`numpy.ndarray`
        obj mesh vertices on which the fingers will be placed
    normals : 2x3 :obj:`numpy.ndarray`
        obj mesh normals at the contact points
    num_facets : int
        number of vectors to use to approximate the friction cone.  these vectors 
        will be along the friction cone boundary
    mu : float 
        coefficient of friction
    gamma : float
        torsional friction coefficient
    object_mass : float
        mass of the object
    mesh : :obj:`Trimesh`
        mesh object

    Returns
    -------
    float : 1 or 0 if the grasp is/isn't force closure for the object
    """
    # fric_cone_angle = np.pi/2 - np.tan(mu)
    # intersect_line = vertices[0] - vertices[1]
    # intersect_line *= np.sign(intersect_line[2]) # make sure f3 is positive
    # unit_intersect_line = intersect_line / np.linalg.norm(intersect_line)
    # for normal in normals:
    #     unit_normal = normal / np.linalg.norm(normal)
    #     angle = np.min(np.arccos(np.dot(unit_normal, unit_intersect_line)), np.arccos(np.dot(unit_normal, -unit_intersect_line)))
    #     if angle < fric_cone_angle or angle == 0:
    #         return 0
    # return 1
    normal0 = -1.0 * normals[0] / (1.0 * np.linalg.norm(normals[0]))
    normal1 = -1.0 * normals[1] / (1.0 * np.linalg.norm(normals[1]))

    alpha = np.arctan(mu)
    line = vertices[0] - vertices[1]
    line = line / (1.0 * np.linalg.norm(line))
    angle1 = np.arccos(normal1.dot(line))

    line = -1 * line
    angle2 = np.arccos(normal0.dot(line))

    if angle1 > alpha or angle2 > alpha:
        return 0
    if gamma == 0:
        return 0
    return 1

def get_grasp_map(vertices, normals, num_facets, mu, gamma):
    """ 
    Defined in the book on page 219. Compute the grasp map given the contact
    points and their surface normals

    Parameters
    ----------
    vertices : 2x3 :obj:`numpy.ndarray`
        obj mesh vertices on which the fingers will be placed
    normals : 2x3 :obj:`numpy.ndarray`
        obj mesh normals at the contact points
    num_facets : int
        number of vectors to use to approximate the friction cone.  these vectors 
        will be along the friction cone boundary
    mu : float 
        coefficient of friction
    gamma : float
        torsional friction coefficient

    Returns
    -------
    6x8 :obj:`numpy.ndarray` : grasp map
    """
    g_1 = look_at_general(vertices[0], normals[0])
    adj_g1 = adj(np.linalg.inv(g_1))
    r_1, p_1 = g_1[:3, :3], g_1[:3, 3]
    g_2 = look_at_general(vertices[1], normals[1])
    adj_g2 = adj(np.linalg.inv(g_2))
    r_2, p_2 = g_2[:3, :3], g_1[:3, 3]
    B = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 1]])
    G_1 = np.matmul(adj_g1.T, B)
    G_2 = np.matmul(adj_g2.T, B) 
    return np.hstack([G_1, G_2])

def find_contact_forces(vertices, normals, num_facets, mu, gamma, desired_wrench):
    """
    Compute that contact forces needed to produce the desired wrench

    Parameters
    ----------
    vertices : 2x3 :obj:`numpy.ndarray`
        obj mesh vertices on which the fingers will be placed
    normals : 2x3 :obj:`numpy.ndarray`
        obj mesh normals at the contact points
    num_facets : int
        number of vectors to use to approximate the friction cone.  these vectors 
        will be along the friction cone boundary
    mu : float 
        coefficient of friction
    gamma : float
        torsional friction coefficient
    desired_wrench : 6x :obj:`numpy.ndarray` potential wrench to be produced

    Returns
    -------
    bool: whether contact forces can produce the desired_wrench on the object
    """

    contact_forces = []
    G = get_grasp_map(vertices, normals, num_facets, mu, gamma)
    opti = Opti()
    f = opti.variable(8, 1)
    alpha = opti.variable(2*(num_facets+1), 1)
    fric_cone = [np.array([0, 0, 1])]
    constraints = []
    general_transform1 = look_at_general(vertices[0], normals[0])
    general_transform2 = look_at_general(vertices[1], normals[1])
    for i in range(num_facets):
        phi = 2 * np.pi * i / num_facets
        fric_cone.append(np.array([np.cos(phi), np.sin(phi), mu]))
    fric_cone = np.stack(fric_cone) # fric_cone is num_facets+1 x 3
    #print(general_transform1.shape, np.matmul(general_transform1[:3, :3], fric_cone).shape)
    #fric_cone = np.vstack([np.matmul(fric_cone, general_transform1[:3, :3]) + general_transform1[:3, 3], np.matmul(fric_cone, general_transform2[:3, :3]) + general_transform2[:3, 3]]) # fric_cone is (2 * num_facets+1) x 3
    # print(fric_cone)
    fric_cone = np.vstack([fric_cone, fric_cone])
    constraints.append(f[:3] == mtimes(fric_cone[:33].T, alpha[:33]))
    constraints.append(f[4:7] == mtimes(fric_cone[33:].T, alpha[33:]))
    constraints.append(f[3] <= gamma * f[2])
    constraints.append(-f[3] <= gamma * f[2])
    constraints.append(f[7] <= gamma * f[6])
    constraints.append(-f[7] <= gamma * f[6])
    constraints.append(alpha >= 0)
    
    constraints.append(mtimes(G, f) == desired_wrench)# negative ?????????????
    obj = mtimes(f.T, f)
    opti.minimize(obj)
    opti.subject_to(constraints)
    opti.set_initial(f, np.zeros([8, 1]))
    opti.set_initial(alpha, np.ones([2*(num_facets+1), 1]))

    opti.solver('ipopt')
    p_opts = {'expand': False}
    s_opts = {'max_iter': 1e4}
    opti.solver('ipopt', p_opts, s_opts)
    try:
        sol = opti.solve()
        contact_force = sol.value(f)
        print("YEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
    except:
        print("CRIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII")
        return None
    return contact_force

def compute_gravity_resistance(vertices, normals, num_facets, mu, gamma, object_mass, mesh):
    """
    Gravity produces some wrench on your object. Computes how much normal force is required
    to resist the wrench produced by gravity.

    Parameters
    ----------
    vertices : 2x3 :obj:`numpy.ndarray`
        obj mesh vertices on which the fingers will be placed
    normals : 2x3 :obj:`numpy.ndarray`
        obj mesh normals at the contact points
    num_facets : int
        number of vectors to use to approximate the friction cone.  these vectors 
        will be along the friction cone boundary
    mu : float 
        coefficient of friction
    gamma : float
        torsional friction coefficient
    object_mass : float
        mass of the object
    mesh : :obj:`Trimesh`
        mesh object

    Returns
    -------
    float: quality of the grasp
    """
    f_g = object_mass * 9.81
    desired_wrench = np.array([0, 0, -f_g, 0, 0, 0])
    
    contact_force = find_contact_forces(vertices, normals, num_facets, mu, gamma, desired_wrench)
    if contact_force is None:
        return 0
    print(contact_force)
    quality = contact_force[3] + contact_force[6]
    return quality

def compute_robust_force_closure(vertices, normals, num_facets, mu, gamma, object_mass, mesh):
    """
    Should return a score for the grasp according to the robust force closure metric.

    Parameters
    ----------
    vertices : 2x3 :obj:`numpy.ndarray`
        obj mesh vertices on which the fingers will be placed
    normals : 2x3 :obj:`numpy.ndarray`
        obj mesh normals at the contact points
    num_facets : int
        number of vectors to use to approximate the friction cone.  these vectors 
        will be along the friction cone boundary
    mu : float 
        coefficient of friction
    gamma : float
        torsional friction coefficient
    object_mass : float
        mass of the object
    mesh : :obj:`Trimesh`
        mesh object

    Returns
    -------
    float: quality of the grasp
    """
    num_monte_carlo = 100
    scale = 0.1
    quality = float(compute_force_closure(vertices, normals, num_facets, mu, gamma, object_mass, mesh)) / num_monte_carlo
    for _ in range(num_monte_carlo):
        new_vertices = []
        for vertex, normal in zip(vertices, normals):
            new_vertex = np.zeros_like(vertex) + vertex
            x_dist = np.random.normal(scale=scale)
            y_dist = np.random.normal(scale=scale)
            z_dist = np.random.normal(scale=scale)
            new_vertex[0] += x_dist
            new_vertex[1] += y_dist
            new_vertex[2] += z_dist
            new_vertices.append(new_vertex)
        on_segments, mesh_nums = find_intersections(mesh, new_vertices[0], new_vertices[1])
        if len(on_segments) != 2:
            continue
        normals = [normal_at_point(mesh, pt) for pt in on_segments]
        quality += float(compute_force_closure(on_segments, normals, num_facets, mu, gamma, object_mass, mesh)) / num_monte_carlo
    return quality

def compute_ferrari_canny(vertices, normals, num_facets, mu, gamma, object_mass, mesh):
    """
    Should return a score for the grasp according to the Ferrari Canny metric.
    Use your favourite python convex optimization package. We suggest casadi.

    Parameters
    ----------
    vertices : 2x3 :obj:`numpy.ndarray`
        obj mesh vertices on which the fingers will be placed
    normals : 2x3 :obj:`numpy.ndarray`
        obj mesh normals at the contact points
    num_facets : int
        number of vectors to use to approximate the friction cone.  these vectors 
        will be along the friction cone boundary
    mu : float 
        coefficient of friction
    gamma : float
        torsional friction coefficient
    object_mass : float
        mass of the object
    mesh : :obj:`Trimesh`
        mesh object

    Returns
    -------
    float: quality of the grasp
    """
    num_monte_carlo = 10
    min_quality = 100000000000000000000
    found_one = False
    for _ in range(num_monte_carlo):
        desired_wrench = np.random.uniform(low=-1, high=1, size=[6])
        desired_wrench /= np.linalg.norm(desired_wrench)
        force = find_contact_forces(vertices, normals, num_facets, mu, gamma, desired_wrench)
        if force is None:
            local_quality = 100000000000000000000
        else:
            found_one=True
            local_quality = 1 / np.sqrt(np.linalg.norm(force))
        min_quality = min(min_quality, local_quality)
    if found_one:
        return min_quality
    else:
        return 0

