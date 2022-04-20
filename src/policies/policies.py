#!/usr/bin/env python -W ignore::DeprecationWarning
"""
Grasping Policy for EE106B grasp planning lab
Author: Chris Correa
"""
import numpy as np
import trimesh

# 106B lab imports
from metrics import (
    compute_force_closure, 
    compute_gravity_resistance,
    compute_robust_force_closure,
    compute_ferrari_canny
)
from utils import length, normalize, find_intersections, look_at_general
from scipy.spatial.transform import Rotation
import vedo

# Can edit to make grasp point selection more/less restrictive
# Based on real-world distances
MAX_GRIPPER_DIST = .075
MIN_GRIPPER_DIST = .03
GRIPPER_LENGTH = 0.105

# These have not been measured, but should still work
CONTACT_MU = 0.5
CONTACT_GAMMA = 0.1
OBJECT_MASS = {'nozzle': .25, 'pawn': .25, 'cube': .25}


class GraspingPolicy():
    def __init__(self, n_vert, n_grasps, n_execute, n_facets, metric_name):
        """
        Parameters
        ----------
        n_vert : int
            We are sampling vertices on the surface of the object, and will use pairs of 
            these vertices as grasp candidates
        n_grasps : int
            how many grasps to sample.  Each grasp is a pair of vertices
        n_execute : int
            how many grasps to return in policy.action()
        n_facets : int
            how many facets should be used to approximate the friction cone between the 
            finger and the object
        metric_name : string
            name of one of the function in src/metrics/metrics.py
        """
        self.n_vert = n_vert
        self.n_grasps = n_grasps
        self.n_execute = n_execute
        self.n_facets = n_facets
        # This is a function, one of the functions in src/metrics/metrics.py
        self.metric = eval(metric_name)

    def vertices_to_baxter_hand_pose(self, vertices, object_mesh):
        """
        Write your own grasp planning algorithm! You will take as input the mesh
        of an object, and a pair of contact points from the surface of the mesh.
        You should return a 4x4 ridig transform specifying the desired pose of the
        end-effector (the gripper tip) that you would like the gripper to be at
        before closing in order to execute your grasp.

        You should be prepared to handle malformed grasps. Return None if no
        good grasp is possible with the provided pair of contact points.
        Keep in mind the constraints of the gripper (length, minimum and maximum
        distance between fingers, etc) when picking a good pose, and also keep in
        mind limitations of the robot (can the robot approach a grasp from the inside
        of the mesh? How about from below?). You should also make sure that the robot
        can successfully make contact with the given contact points without colliding
        with the mesh.

        The trimesh package has several useful functions that allow you to check for
        collisions between meshes and rays, between meshes and other meshes, etc, which
        you may want to use to make sure your grasp is not in collision with the mesh.

        Take a look at the functions find_intersections, find_grasp_vertices, 
        normal_at_point in utils.py for examples of how you might use these trimesh 
        utilities. Be wary of using these functions directly. While they will probably 
        work, they don't do excessive edge-case handling. You should spend some time
        reading the documentation of these packages to find other useful utilities.
        You may also find the collision, proximity, and intersections modules of trimesh
        useful.

        Feel free to change the signature of this function to add more arguments
        if you believe they will be useful to your planner. We've provided some starter
        code to give you some structure for this function, but feel free to ignore this
        as well.

        Parameters
        ----------
        object_mesh (trimesh.base.Trimesh): A triangular mesh of the object, as loaded in with trimesh.
        vertices (2x3 np.ndarray): obj mesh vertices on which the fingers will be placed

        Returns
        -------
        (4x4 np.ndarray): The rigid transform for the desired pose of the gripper, in the object's reference frame.
        """
        # HINT: `look_at_general` in utils.py might be a nice starting point
        # origin of gripper pose when grasping
        origin = (vertices[0] + vertices[1] ) / 2
        print(origin)

        # Potential orientation of gripper
        up = np.array([0, 0, 1])
        init_transform = look_at_general(origin, -up) # ????????????????????????????????????
        direction = vertices[0] - vertices[1]
        y = normalize(direction) # create a z vector in the given direction
        x = normalize(np.cross(up, y)) # create a x vector perpendicular to z and up
        z = np.cross(y, x) # create a y vector perpendicular to z and x
        #why they in this order?????????????????????????????????
        init_y = x
        init_x = y
        init_z = z

        rot_y = lambda theta: np.array([[np.cos(theta), 0, np.sin(theta)],
                                        [0, 1, 0],
                                        [-np.sin(theta), 0, np.cos(theta)]])

        # The current gripper orientation might not be feasible
        # One strategy is to see if rotating the gripper produces a feasible orientation

        # Explore orientations in order of closeness from starting orientation
        n_orientations = 10
        #lin = np.linspace(0, np.pi/2, n_orientations)
        lin = np.linspace(0, np.pi/2, n_orientations)
        iter_list = []
        for i in lin:
            iter_list.append(i)
            iter_list.append(-i)

        for rot_val in iter_list:
            # Determine new gripper oreintation when rotating gripper around y axis
            roty_matrix = rot_y(rot_val)
            #why they in this order?????????????????????????????????
            matrix = np.stack([init_x, init_y, init_z]).T
            rotated = np.matmul(roty_matrix, matrix)
            y = rotated[:3, 0]
            x = rotated[:3, 1]
            z = rotated[:3, 2]
            gripper_top = origin - GRIPPER_LENGTH * z
            gripper_double = origin - 2 * GRIPPER_LENGTH * z
            feasible = True # MAKE SURE TO ADD SOME CHECKS THAT POTENTIALLY SET THIS TO FALSE

            #check for wrist collisions
            wrist_collisions, _ = find_intersections(object_mesh, gripper_top, gripper_double)
            if len(wrist_collisions) > 0:
                feasible = False
            
            # check for scooping from the bottom
            if z[2] > 0:
                feasible = False
            # Here, you will want to set feasible to 'False' if the current grippper orientation
            # is infeasible. This will mostly be collision checking; take a look at the docstring
            # of this function for ideas on this.


            if feasible:
                result = np.eye(4)
                result[0:3,0] = x
                result[0:3,1] = y
                result[0:3,2] = z
                result[0:3,3] = origin
                return result

        return None



    def sample_grasps(self, vertices, normals):
        """
        Samples a bunch of candidate grasps points. You should randomly choose pairs of vertices
        and throw out pairs which are too big for the gripper, or too close too the table. 
        You may want to throw out vertices which are lower than ~3cm of the table. 
        Returns the pairs of grasp vertices and grasp normals (the normals at the grasp vertices).

        Parameters
        ----------
        vertices : nx3 :obj:`numpy.ndarray`
            mesh vertices
        normals : nx3 :obj:`numpy.ndarray`
            mesh normals

        Returns
        -------
        n_graspsx2x3 :obj:`numpy.ndarray`
            grasps vertices.  Each grasp containts two contact points.  Each contact point
            is a 3 dimensional vector and there are n_grasps of them, hence the shape n_graspsx2x3
        n_graspsx2x3 :obj:`numpy.ndarray`
            grasps normals.  Each grasp containts two contact points.  Each vertex normal
            is a 3 dimensional vector, and there are n_grasps of them, hence the shape n_graspsx2x3
        """
        ret_vertices = []
        ret_normals = []
        table_height = np.min(vertices[:, -1])
        while len(ret_vertices) < self.n_grasps:
            print("SAMPLING GRASOPS", len(ret_vertices))
            idx1 = np.random.randint(0, len(vertices))
            idx2 = np.random.randint(0, len(vertices))
            vert1 = vertices[idx1]
            vert2 = vertices[idx2]
            print("checking vertices:", vert1, vert2)
            if vert1[2] <= table_height + 0.015: # this z-coordinate check is in base/world frame! was -0.14 before was 0.03 before
                print("VERTEX 1 TOO LOW")
                continue
            if vert2[2] <= table_height + 0.015:
                print("VERTEX 2 TOO LOW")
                continue
            dist = np.linalg.norm(vert1 - vert2)
            if dist > MAX_GRIPPER_DIST:
                print("TOO WIDE", dist)
                continue
            if dist < MIN_GRIPPER_DIST:
                print("too narrow", dist)
                continue
            ret_vertices.append(np.stack([vert1, vert2]))
            ret_normals.append(np.stack([normals[idx1], normals[idx2]]))
            
        return np.stack(ret_vertices), np.stack(ret_normals)

    def score_grasps(self, grasp_vertices, grasp_normals, object_mass, mesh):
        """
        Takes mesh and returns pairs of contacts and the quality of grasp between the contacts, sorted by quality
        
        Parameters
        ----------
        grasp_vertices : n_graspsx2x3 :obj:`numpy.ndarray`
            grasps.  Each grasp containts two contact points.  Each contact point
            is a 3 dimensional vector, and there are n_grasps of them, hence the shape n_graspsx2x3
        grasp_normals : mx2x3 :obj:`numpy.ndarray`
            grasps normals.  Each grasp containts two contact points.  Each vertex normal
            is a 3 dimensional vector, and there are n_grasps of them, hence the shape n_graspsx2x3

        Returns
        -------
        :obj:`list` of int
            grasp quality for each 
        """
        scores = []
        for i in range(len(grasp_vertices)):
            
            scores.append(self.metric(grasp_vertices[i], grasp_normals[i], self.n_facets, CONTACT_MU, CONTACT_GAMMA, object_mass, mesh))
            print("scoring vertices:", i, scores[-1])
        scores = np.array(scores).astype(np.float64)
        # scores = scores - np.min(scores)
        # scores /= np.max(scores)
        return scores


    def vis(self, mesh, grasp_vertices, grasp_qualities):
        """
        Pass in any grasp and its associated grasp quality.  this function will plot
        each grasp on the object and plot the grasps as a bar between the points, with
        colored dots on the line endpoints representing the grasp quality associated
        with each grasp
        
        Parameters
        ----------
        mesh : :obj:`Trimesh`
        grasp_vertices : mx2x3 :obj:`numpy.ndarray`
            m grasps.  Each grasp containts two contact points.  Each contact point
            is a 3 dimensional vector, hence the shape mx2x3
        grasp_qualities : mx' :obj:`numpy.ndarray`
            vector of grasp qualities for each grasp
        """
        
        vis3d.mesh(mesh)
        print("VIZ")
        dirs = normalize(grasp_vertices[:,0] - grasp_vertices[:,1], axis=1)
        midpoints = (grasp_vertices[:,0] + grasp_vertices[:,1]) / 2
        grasp_vertices[:,0] = midpoints + dirs*MAX_GRIPPER_DIST/2
        grasp_vertices[:,1] = midpoints - dirs*MAX_GRIPPER_DIST/2

        for grasp, quality in zip(grasp_vertices, grasp_qualities):
            color = [min(1, 2*(1-quality)), min(1, 2*quality), 0, 1]
            vis3d.plot3d(grasp, color=color, tube_radius=.001)
        vis3d.show()
        
    def visualize_grasp(self, mesh, vertices, pose):
        """Visualizes a grasp on an object. Object specified by a mesh, as
        loaded by trimesh. vertices is a pair of (x, y, z) contact points.
        pose is the pose of the gripper tip.
        Parameters
        ----------
        mesh (trimesh.base.Trimesh): mesh of the object
        vertices (np.ndarray): 2x3 matrix, coordinates of the 2 contact points
        pose (np.ndarray): 4x4 homogenous transform matrix
        """
        p1, p2 = vertices
        center = (p1 + p2) / 2
        approach = pose[:3, 2]
        tail = center - GRIPPER_LENGTH * approach

        contact_points = []
        for v in vertices:
            contact_points.append(vedo.Point(pos=v, r=30))

        vec = (p1 - p2) / np.linalg.norm(p1 - p2)
        line = vedo.shapes.Tube([center + 0.5 * MAX_GRIPPER_DIST * vec,
                                       center - 0.5 * MAX_GRIPPER_DIST * vec], r=0.001, c='g')
        approach = vedo.shapes.Tube([center, tail], r=0.001, c='g')
        vedo.show([mesh, line, approach] + contact_points, new=True)

    def top_n_actions(self, mesh, obj_name, vis=True):
        """
        Takes in a mesh, samples a bunch of grasps on the mesh, evaluates them using the 
        metric given in the constructor, and returns the best grasps for the mesh.  SHOULD
        RETURN GRASPS IN ORDER OF THEIR GRASP QUALITY.

        Parameters
        ----------
        mesh : :obj:`Trimesh`
        vis : bool
            Whether or not to visualize the top grasps

        Returns
        -------
        :obj:`list` of :obj:Pose
            the matrices T_world_grasp, which represents the hand poses of the baxter / sawyer
            which would result in the fingers being placed at the vertices of the best grasps

        RETURNS LIST OF LISTS
        """
        # Some objects have vertices in odd places, so you should sample evenly across 
        # the mesh to get nicer candidate grasp points using trimesh.sample.sample_surface_even()
        all_vertices, all_poses = [], []


        while len(all_vertices) < self.n_execute:
            assert len(all_vertices) == len(all_poses)
            vertices, face_ind = trimesh.sample.sample_surface_even(mesh, self.n_vert)
            print("object vertices", vertices)
            normals = mesh.face_normals[face_ind]

            grasp_vertices, grasp_normals = self.sample_grasps(vertices, normals)
            mass = OBJECT_MASS[obj_name]
            grasp_qualities = self.score_grasps(grasp_vertices, grasp_normals, mass, mesh)
            

            # This is the vertices of the grasps with the highest grasp qualities. Should be shape:
            # n_executex2x3
            top_vertex_ind = np.argsort(-grasp_qualities)[:self.n_execute]
            top_n_vertices = grasp_vertices[top_vertex_ind]
            top_n_scores = grasp_qualities[top_vertex_ind]
            print("GRASP VERTICES ", top_n_vertices)
            print("GRASP SCORES", top_n_scores)

            poses = [self.vertices_to_baxter_hand_pose(vertices, mesh) for vertices in top_n_vertices]
            left_vertices = [v for (v, p) in zip(top_n_vertices, poses) if p is not None]
            poses = [p for p in poses if p is not None]

            all_vertices.extend(left_vertices)
            all_poses.extend(poses)

        return all_vertices, all_poses
