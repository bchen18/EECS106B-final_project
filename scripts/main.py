#!/usr/bin/env python
"""
Starter Script for C106B Grasp Planning Lab
Authors: Chris Correa, Riddhi Bagadiaa, Jay Monga
"""
from readline import get_completion_type
from cv2 import transform
import numpy as np
import cv2
import argparse
from scipy.spatial.transform import Rotation
from rospy import Publisher, ROSException
from utils import rotation_from_quaternion, create_transform_matrix, quaternion_from_matrix
import trimesh
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans
from policies import GraspingPolicy
import time
from visualization_msgs.msg import Marker, MarkerArray

from copy import deepcopy
import vedo
try:
    import rospy
    import tf
    from cv_bridge import CvBridge
    from geometry_msgs.msg import Pose, PoseStamped
    import sensor_msgs.point_cloud2
    from sensor_msgs.msg import Image, CameraInfo, PointCloud2
    from std_msgs.msg import Float32
    from baxter_interface import gripper as baxter_gripper
    from intera_interface import gripper as sawyer_gripper
    from path_planner import PathPlanner
    ros_enabled = True
except:
    print('Couldn\'t import ROS.  I assume you\'re running this on your laptop')
    ros_enabled = False

class ParticleFilter:
    def __init__(self, num_particles, initial_guess, sample_std, mesh_fn = lambda pose: trimesh.primitives.Cylinder(radius=0.0762/2, height=0.12065, transform=pose), min_z=0):
        self.num_particles = num_particles
        self.initial_guess = initial_guess
        self.mesh_fn = mesh_fn
        self.sample_std = sample_std
        # create initial particle
        self.particles = [mesh_fn(self.initial_guess)]
        self.weights = [1]
        self.min_z=min_z
        # sample initial set of particles from initial guess
        self.particles = self.create_particles()
        self.weights = [1.0/self.num_particles for _ in range(self.num_particles)]
        print("INITIAL WEIGHTS: ", self.weights)
        self.markerArray = []
        
    
    def create_particles(self):
        particles = []
        for _ in range(self.num_particles):
            base_particle = np.random.choice(self.particles, p = self.weights)
            sampled_particle = self.sample_particle(base_particle)
            particles.append(sampled_particle)
        return particles

    def sample_particles(self):
        self.particles = self.create_particles()
        self.weights = [1.0/self.num_particles for _ in range(self.num_particles)]

    def sample_particle(self, base_particle):
        # generate a uniformly random rotation
        rand_rot_perturb, _ = np.linalg.qr(np.random.randn(3, 3), mode='complete')
        rand_pos_perturb = np.random.normal(scale=self.sample_std, size=[3])
        curr_transform = deepcopy(base_particle.primitive.transform)
        curr_transform[:3, :3] = np.matmul(curr_transform[:3, :3], rand_rot_perturb)
        curr_transform[:3, -1] += rand_pos_perturb
        curr_transform[2, -1] = max(self.min_z, curr_transform[2, -1])
        new_particle = self.mesh_fn(curr_transform)
        return new_particle

    def reweight_particle(self, particle_idx, mult_factor):
        self.weights[particle_idx] *= mult_factor
    
    def renormalize_weights(self):
        total_weight = sum(self.weights)
        if total_weight == 0:
            total_weight = 1
        #renormalize weights
        self.weights = [p / total_weight for p in self.weights]

    def sorted_particles(self):
        idxs = np.argsort(self.weights)
        print("idxs: ", idxs)
        sorted_particles = [self.particles[idx] for idx in idxs][::-1]
        return sorted_particles
    
    def center_particle(self):
        best_particle = self.sorted_particles()[0]
        mean_pos = np.array([0.0,0.0,0.0])
        for i, particle in enumerate(self.particles):
            mean_pos += self.weights[i] * particle.primitive.transform[:3, -1]
        new_pose = np.zeros([4,4])
        new_pose[:3, :3] = best_particle.primitive.transform[:3, :3]
        new_pose[:3, -1] = mean_pos
        new_pose[3, 3] =  1
        return self.mesh_fn(new_pose)

    def reweight_particles(self, touch_pose, got_touched):
        touch_position = touch_pose[:3, -1]
        touch_rot = touch_pose[:3, :3]
        for i, particle in enumerate(self.particles):
            point, dist, _ = trimesh.proximity.closest_point(particle, [touch_position])
            dist = dist[0]
            print("DIST", dist)
            if got_touched:
                factor = (1/dist) ** 0.2
            else:
                factor = dist ** 0.2
            print("PARTICLE SIZE: ", len(self.particles), self.weights)
            self.reweight_particle(i, factor)
        self.renormalize_weights()

    def makeMarkers(self):
        for i, particle in enumerate(self.particles):
            position = particle.primitive.transform[:3, -1]
            orientation = quaternion_from_matrix(particle.primitive.transform[:3, :3])
            marker = Marker()
            marker.header.frame_id = "base"
            marker.type = 1
            marker.id = i
            marker.action = 0
            marker.pose.position.x = position[0]
            marker.pose.position.y = position[1]
            marker.pose.position.z = position[2]
            marker.pose.orientation.x = orientation[0]
            marker.pose.orientation.y = orientation[1]
            marker.pose.orientation.z = orientation[2]
            marker.pose.orientation.w = orientation[3]
            marker.color.r=1
            marker.color.a=0.6
            marker.scale.x=0.1
            marker.scale.y=0.1
            marker.scale.z=0.1
            self.markerArray.append(marker)
        markerArray = MarkerArray()
        markerArray.markers = self.markerArray
        return markerArray          





def lookup_transform(to_frame, from_frame='base', no_swap = False):
    """
    Returns the AR tag position in world coordinates 

    Parameters
    ----------
    to_frame : string
        examples are: ar_marker_7, nozzle, pawn, ar_marker_3, etc
    from_frame : string
        lets be real, you're probably only going to use 'base'

    Returns
    -------
    :4x4 :obj:`numpy.ndarray` relative pose between frames
    """
    if not ros_enabled:
        print('I am the lookup transform function!  ' \
            + 'You\'re not using ROS, so I\'m returning the Identity Matrix.')
        return np.identity(4)
    listener = tf.TransformListener()
    attempts, max_attempts, rate = 0, 500, rospy.Rate(0.1)
    tag_rot=[]
    print("entering")
    while attempts < max_attempts:
        try:
            t = listener.getLatestCommonTime(from_frame, to_frame)
            tag_pos, tag_rot = listener.lookupTransform(from_frame, to_frame, t)
            attempts = max_attempts
        except:
            print("exception!")
            rate.sleep()
            attempts += 1
    print("exiting")
    if no_swap:
        r = Rotation.from_quat(tag_rot)
        try:
            rot = r.as_dcm()
        except:
            rot = r.as_matrix()
    else:
        rot = rotation_from_quaternion(tag_rot)
    return create_transform_matrix(rot, tag_pos)


def rot_z(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta)],
                     [np.sin(theta), np.cos(theta), 0], 
                     [0, 0, 1]])

# Use multiview reconstruction to get point cloud and initial mesh guess. This doesn't work!
def do_multiview(camera_image_topic, camera_info_topic, camera_frame, planner, gripper):
    curr_pose = lookup_transform("right_gripper_tip", no_swap=True)[:3, -1]
    bridge = CvBridge()

    # Get first object image
    moveit_plan(curr_pose+np.array([0, 0, 0]), np.array([0, 1, 0, 0]), speed=0.2)
    rospy.sleep(0.7)
    image1 = rospy.wait_for_message(camera_image_topic, Image)
    cv_image1 = bridge.imgmsg_to_cv2(image1, desired_encoding='bgr8')#used passthrough
    info = rospy.wait_for_message(camera_info_topic, CameraInfo)
    T_world_camera_before = lookup_transform(to_frame = camera_frame, from_frame="base", no_swap=True)
    cv2.imwrite("im1.png", cv_image1)
    #cv_image1 = np.uint8(cv_image1) #need to cast to uint8 for StereoBM_create?
    #cv_image1 = cv2.cvtColor(cv_image1, cv2.COLOR_BGR2GRAY)

    # Move a bit, then get second object image
    #moveit_plan(curr_pose, np.array([1, 0, 0, 0]), speed=1.0)
    moveit_plan(curr_pose+np.array([0, -0.1, 0]), np.array([0, 1, 0, 0]), speed=0.2)
    rospy.sleep(0.7)
    T_world_camera_after = lookup_transform(to_frame = camera_frame, from_frame="base", no_swap=True)
    image2 = rospy.wait_for_message(camera_image_topic, Image)
    cv_image2 = bridge.imgmsg_to_cv2(image2, desired_encoding='bgr8')#used passthrough
    cv2.imwrite("im2.png", cv_image2)
    #cv_image2 = np.uint8(cv_image2) #need to cast to uint8 for StereoBM_create?
    #cv_image2 = cv2.cvtColor(cv_image2, cv2.COLOR_BGR2GRAY)

    # Test images
    # cv_image1 = cv2.imread('ambush_5_left.jpg')
    # cv_image2 = cv2.imread('ambush_5_right.jpg')

    # Construct disparity map
    #stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    window_size = 3
    min_disp = 16
    num_disp = 112-min_disp
    # having just blockSize=15 works decent
    stereo = cv2.StereoSGBM_create(
            # numDisparities=16,#112,
            #minDisparity = min_disp,
            #numDisparities = num_disp,
             blockSize = 15,#15,
            #P1 = 8*3*window_size**2,
            #P2 = 32*3*window_size**2,
            #disp12MaxDiff = 0,
            #uniquenessRatio = 10,
            #speckleWindowSize = 10,
            #speckleRange = 1
        )
    disp = stereo.compute(cv_image1, cv_image2)
    print("disparity:", disp)

    # Try some post-processing on disparity map
    #disp = cv2.fastNlMeansDenoising(disp, None)
    cv2.imwrite("disp.png", disp)
    #return

    # Grab Q matrix
    h, w = cv_image1.shape[:2]
    f = 0.8*w                          # guess for focal length
    K = np.array(info.K).reshape([3,3])
    dist = info.D
    #R = rot_z(np.pi)
    R = rot_z(0)
    T = T_world_camera_after[:3, 3] - T_world_camera_before[:3, 3]
    _, _, _, _, Q, _, _ = cv2.stereoRectify(K, dist, K, dist, (w, h), R, T)# reverse the order of h and w??
    # WHY IS Q MATRIX 4x4 instead of 3x3????
    # Q = np.float32([[1, 0, 0, -0.5*w],
    #             [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
    #             [0, 0, 0,     -f], # so that y-axis looks up
    #             [0, 0, 1,      0]])

    # Reproject image points to 3D using disparity map and Q matrix
    points = cv2.reprojectImageTo3D(disp, Q)
    pts = np.array(points).reshape([-1, 3])
    print(pts.shape)
    # remove points above 0
    pts = pts[pts[:,2]<0]
    print(pts.shape)
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter(pts[::15, 0], pts[::15, 1], pts[::15, 2])
    #ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2])
    plt.show()
    colors = cv2.cvtColor(cv_image1, cv2.COLOR_BGR2RGB)
    mask = disp > disp.min()
    out_points = points[mask]
    out_colors = colors[mask]

# Use RealSense IR camera to get pointcloud and initial mesh guess
def realsense_pointcloud(realsense_topic, camera_frame, use_kmeans=True, obj_shape='cube'):
    print("getting realsense pointcloud")
    # Get pointcloud from realsense camera. Points are in camera frame!
    pointcloud = rospy.wait_for_message(realsense_topic, PointCloud2)
    points_3d = np.array([])
    for point in sensor_msgs.point_cloud2.read_points(pointcloud, skip_nans=True):
            if points_3d.size == 0:
                points_3d = np.array(point[:3]).reshape((1, 3))
            else:
                curr_point = np.array(point[:3]).reshape((1, 3))
                points_3d = np.vstack((points_3d, curr_point))

    print(points_3d.shape) #should be N x 3

    # Filter pointcloud to remove noise and table points
    points_3d = points_3d[points_3d[:, 2]<0.75] # anything greater than 0.75 meter away from camera is noise
    points_3d = points_3d[points_3d[:, 2]>0.2] # anything less than 0.2 meter from camera is also noise

    # Transform points in pointcloud to base frame
    camera_pose = lookup_transform(camera_frame, no_swap=True)
    homog = np.ones([points_3d.shape[0], 1])
    points_3d = np.matmul(camera_pose, np.concatenate([points_3d, homog], axis=-1).T ).T

    # Get "lowest" point which corresponds to table z coordinate, then remove points slightly above that to remove the table
    # this is trucky, may need a different way to remove the table
    median_z = np.mean(points_3d[:, 2])
    lowest_point = np.argmin(points_3d[:, 2])
    highest_point = np.argmax(points_3d[:, 2])
    lowest_z = points_3d[lowest_point, 2]
    print(lowest_z)
    # fig = plt.figure()
    # ax = plt.axes(projection="3d")
    # ax.scatter(points_3d [::10, 0], points_3d[::10, 1], points_3d[::10, 2])
    # plt.show()
    if use_kmeans:
        # Try filtering out table with k means
        table_detector = KMeans(4)
        point_labels = table_detector.fit_predict(points_3d[:, 2].reshape([-1, 1]))
        highest_label = point_labels[highest_point]
        points_3d = points_3d[point_labels == highest_label]
    else:
        
        points_3d = points_3d[points_3d[:, 2]<lowest_z+0.05] # anything greater than table height away from camera is probably the table
    
    # Show pointcloud
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter(points_3d [::10, 0], points_3d[::10, 1], points_3d[::10, 2])
    plt.show()
    
    

    
    lowest_point = np.argmin(points_3d[:, 2])
    lowest_z = points_3d[lowest_point, 2]


    # points_3d = np.matmul(points_3d, camera_pose[:3, :3])
    # points_3d += camera_pose[:3, -1] # may need to do this in loop idk how to broadcast

    centroid = np.mean(points_3d, axis=0)
    print(centroid)
    R = np.eye(3) # dummy initial rotation

    object=0

    pose = np.zeros([4, 4])
    pose[:3, :3] = R
    pose[3, 3] = 1
    pose[:3, -1] = centroid[:3]

    # Create trimesh object (we know object shape beforehand)
    if obj_shape == "cube":
        #return early for now  
        side_length = 4.9
        object = trimesh.primitives.Box(extents=(side_length, side_length, side_length), transform=pose)
    elif obj_shape == "realsense":
        object = trimesh.primitives.Box(extents=(0.0889, 0.1397, 0.0508), transform=pose)
        mesh_fn = lambda pose: trimesh.primitives.Box(extents=(0.0889, 0.1397, 0.0508), transform=pose)
    elif obj_shape == "plentea":
        object = trimesh.primitives.Box(extents=(0.0635, 0.1778, 0.0635), transform=pose)
        mesh_fn = lambda pose: trimesh.primitives.Box(extents=(0.0635, 0.1778, 0.0635), transform=pose)
    elif obj_shape == "glass":
        object = trimesh.primitives.Cylinder(radius=0.0762/2, height=0.12065, transform=pose)
        mesh_fn = lambda pose: trimesh.primitives.Cylinder(radius=0.0762/2, height=0.12065, transform=pose)
    print(median_z)
    particle_filter = ParticleFilter(10, pose, 0.05, mesh_fn=mesh_fn, min_z=median_z+0.13)
    return particle_filter

# Uses moveit to move to the specified point and orientation (in base frame)
# Inputs:
#   - point: [x, y, z] np array 
#   - orientation: [x, y, z, w] np array as quarternion
# Outputs:
#   - boolean for if it touched something
def moveit_plan(point, orientation, speed=0.1, force_feedback=False):
    planner = PathPlanner('{}_arm'.format("right"))
    pose = Pose()
    pose.position.x = point[0]
    pose.position.y = point[1]
    pose.position.z = point[2]
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = tuple(orientation)
    planner.change_velocity(speed)
    plan = planner.plan_to_pose(pose)
    planner.execute_plan(plan, wait=not force_feedback)
    if force_feedback:
        try:
            last_msg = 204600
            start = time.time()
            curr_time = start
            while last_msg > 30000 and curr_time < start+10:
                last_msg = rospy.wait_for_message('force_resistance', Float32, timeout=10).data
                print("DETECTED POTENTIAL TOUCH")
                curr_time = time.time()
            if curr_time >= start + 10:
                raise ROSException()
            print("DETECTED TOUCH")
            planner._group.stop()
            return True
        except ROSException as e:
            print("NO TOUCH")
            return False
    

def execute_grasp(T_world_grasp, planner, gripper):
    """
    Perform a pick and place procedure for the object. One strategy (which we have
    provided some starter code for) is to
    1. Move the gripper from its starting pose to some distance behind the object
    2. Move the gripper to the grasping pose
    3. Close the gripper
    4. Move up
    5. Place the object somewhere on the table
    6. Open the gripper. 

    As long as your procedure ends up picking up and placing the object somewhere
    else on the table, we consider this a success!

    HINT: We don't require anything fancy for path planning, so using the MoveIt
    API should suffice. Take a look at path_planner.py. The `plan_to_pose` and
    `execute_plan` functions should be useful. If you would like to be fancy,
    you can also explore the `compute_cartesian_path` functionality described in
    http://docs.ros.org/en/kinetic/api/moveit_tutorials/html/doc/move_group_python_interface/move_group_python_interface_tutorial.html
    
    Parameters
    ----------
    T_world_grasp : 4x4 :obj:`numpy.ndarray`
        pose of gripper relative to world frame when grasping object
    """
    def close_gripper():
        """closes the gripper"""
        gripper.close() # HAD BLOCK=TRUE
        rospy.sleep(2.0)

    def open_gripper():
        """opens the gripper"""
        gripper.open() # HAD BLOCK=TRUE
        rospy.sleep(4.0)

    inp = raw_input('Press <Enter> to move, or \'exit\' to exit')
    if inp == "exit":
        return

    # Go behind object
    print("TWORLD GRASP", T_world_grasp[:, 3])
    gripper_direction = T_world_grasp[:3, 0]
    pose = Pose()
    planner.change_velocity(0.3)
    # pose.position.x = T_world_grasp[0, 3] - 0.2
    # pose.position.y = T_world_grasp[1, 3]
    # pose.position.z = T_world_grasp[2, 3] + 0.1
    pose.position.x = T_world_grasp[0, 3] #- gripper_direction[0] * 0.2
    pose.position.y = T_world_grasp[1, 3] #- gripper_direction[1] * 0.2
    pose.position.z = T_world_grasp[2, 3] + 0.4 #- gripper_direction[2] * 0.2
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = quaternion_from_matrix(T_world_grasp[:3, :3])

    q = [0.0, 0.00, -0.34] # warning this is a little lower than the actual table
    w = [0., 0., 0., 1.]
    size = [3., 3., 0.1]
    table = PoseStamped()
    table.header.frame_id = "base"

    table.pose.position.x = q[0]
    table.pose.position.y = q[1]
    table.pose.position.z = q[2]

    table.pose.orientation.x = w[0]
    table.pose.orientation.y = w[1]
    table.pose.orientation.z = w[2]
    table.pose.orientation.w = w[3]

    planner.add_box_obstacle(np.array(size), "table", table)
    for i in range(5):
        try:
            plan = planner.plan_to_pose(pose)
            open_gripper()
            
            planner.execute_plan(plan)
            break
        except:
            print("PAIN1")
    rospy.sleep(1.0)
    
    # Then swoop in
    pose.position.x = T_world_grasp[0, 3] - 0.02
    pose.position.y = T_world_grasp[1, 3] #+ 0.04
    pose.position.z = T_world_grasp[2, 3] + 0.03
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = quaternion_from_matrix(T_world_grasp[:3, :3])
    for i in range(5):
        try:
            plan = planner.plan_to_pose(pose)
            planner.execute_plan(plan)
            break
        except:
            print("PAIN2")
    rospy.sleep(1.0)

    # Bring the object up
    close_gripper()
    pose.position.x = T_world_grasp[0, 3]
    pose.position.y = T_world_grasp[1, 3]
    pose.position.z = T_world_grasp[2, 3] + 0.2
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = quaternion_from_matrix(T_world_grasp[:3, :3])
    for i in range(5):
        try:
            plan = planner.plan_to_pose(pose)
            planner.execute_plan(plan)
            break
        except:
            print("PAIN3")

    # And over
    pose.position.x = T_world_grasp[0, 3]
    pose.position.y = T_world_grasp[1, 3] - 0.1
    pose.position.z = T_world_grasp[2, 3] + 0.2
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = quaternion_from_matrix(T_world_grasp[:3, :3])
    for i in range(5):
        try:
            plan = planner.plan_to_pose(pose)
            planner.execute_plan(plan)
            break
        except:
            print("PAIN4")

    # And now place it
    #planner.execute_plan(plan)
    open_gripper()


def z_rotation(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]])

def cam_to_base(x, info, T_world_camera):
    K = np.array(info.K).reshape([3, 3])
    k_inv = np.linalg.inv(K)
    cam_lambda = 0.6 # height from camera to top of object (should be around 60 cm when starting from tuck.launch) this will correct Z coordinate
    pixel_to_3d = lambda x: np.matmul(k_inv, x)
    center_img_point = np.array(x).reshape((2, 1))
    print("center_img_point:", center_img_point)
    center_world_point = cam_lambda*pixel_to_3d(np.vstack((center_img_point, np.array([1])))) # this is camera to object frame?
    print("center_world_point (cam to obj):", center_world_point)
    center_world_point = np.vstack( (center_world_point, np.array([1])))
    center_world_point = np.matmul(T_world_camera, center_world_point)
    print("center_world_point (base to obj):", center_world_point)
    return center_world_point

def locate_cube(camera_image_topic, camera_info_topic, camera_frame):
    """
    Finds size and pose of cube in field of view.
    We are leaving this very open ended! Feel free to be creative!
    OpenCV will probably be useful. You may want to look for the
    corners of the cube, or its edges. From that, try your best to reconstruct
    the actual size and pose of the cube!

    Parameters
    ----------
    camera_image_topic : string
        ROS topic for camera image
    camera_info_topic : string
        ROS topic for camera info
    camera_frame : string
        Name of TF frame for camera

    Returns
    -------
    :obj:`trimesh.primatives.Box` : trimesh object for reconstructed cube
    """
    bridge = CvBridge()
    image = rospy.wait_for_message(camera_image_topic, Image)
    cv_image = bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')#used passthrough
    info = rospy.wait_for_message(camera_info_topic, CameraInfo)
    T_world_camera = lookup_transform(to_frame = camera_frame, from_frame="base", no_swap=True)
    T_world_camera_copy = np.copy(T_world_camera)

    # Do your image processing on cv_image here!
    color1 = np.array([122, 73, 77]) # purple
    color2 = np.array([125, 70, 49])
    color3 = np.array([60, 93, 70])
    cv2.imwrite("img.png", cv_image)
    img_height = cv_image.shape[0] #corresponds to rows
    img_width = cv_image.shape[1] #columns
   
    # segement image to make corner finding easier
    gray_img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    img = threshold_segment_color(cv_image, color1-35, color1+35) #try to isolate purple
    blur = cv2.blur(img,(5,5))
    blur[blur < 100] = 0
    cv2.imshow('gray_img', blur)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()
    im2,contours,hierarchy = cv2.findContours(blur, 1, 2)

    # get bounding box info "rect" around cube
    rect = cv2.minAreaRect(contours[0])
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    boxed = cv2.drawContours(blur,[box],0, 128,2)

    # draw bounding box and corner points
    for i in box:
        x, y = i.ravel()
        cv2.circle(boxed, (x, y), 3, 128, -1)
    cv2.imshow('boxed', boxed)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

    # convert cube center from image frame to world frame
    center_world_point = cam_to_base(rect[0], info, T_world_camera)
    #fuck it
    center_world_point[0] += 0.05
    center_world_point[1] += 0.05
    corners = []
    for i in box:
        x, y = i.ravel()
        corners.append(cam_to_base([x, y], info, T_world_camera))
    side_length = 10000000000000000000000000000000000000
    for i in range(len(corners)):
        for j in range(i+1, len(corners)):
            side_length = min(side_length, np.linalg.norm(corners[i] - corners[j]))
    #side_length *= 100
    # K = np.array(info.K).reshape([3, 3])
    # k_inv = np.linalg.inv(K)
    # cam_lambda = 0.6 # height from camera to top of object (should be around 60 cm when starting from tuck.launch) this will correct Z coordinate
    # pixel_to_3d = lambda x: np.matmul(k_inv, x)
    # center_img_point = np.array(rect[0]).reshape((2, 1))
    # print("center_img_point:", center_img_point)
    # center_world_point = cam_lambda*pixel_to_3d(np.vstack((center_img_point, np.array([1])))) # this is camera to object frame?
    # print("center_world_point (cam to obj):", center_world_point)
    # center_world_point = np.vstack( (center_world_point, np.array([1])))
    # center_world_point = np.matmul(T_world_camera, center_world_point)
    # print("center_world_point (base to obj):", center_world_point)

    # get cube orientation (only care about its 2D rotation about z axis because its top face is always parallel to xy plane)
    print("rect angle:", rect[2])
    R = z_rotation(rect[2] * np.pi * 2 / 360)

    # get cube side length

    print("CUBE SIDE LENGTH", side_length)
    #TODO: ACTUAL POSE
    #side_length = 4.9 # length of one side of cube
    pose = np.zeros((4, 4)) # 4x4 homogenous transform for center of cube
    pose[:3, :3] = R
    pose[3, 3] = 1
    pose[:3, 3] = center_world_point[:3].reshape([3,])
    #import pdb; pdb.set_trace()
    return trimesh.primitives.Box(extents=(side_length, side_length, side_length), transform=pose)

def threshold_segment_gray(gray_img, lower_thresh, upper_thresh):
    """perform grayscale thresholding using a lower and upper threshold by
    blacking the background lying between the threholds and whitening the
    foreground
    Parameter
    ---------
    gray_img : ndarray
        grayscale image array
    lower_thresh : float or int
        lowerbound to threshold (an intensity value between 0-255)
    upper_thresh : float or int
        upperbound to threshold (an intensity value between 0-255)
    Returns
    -------
    ndarray
        thresholded version of gray_img
    """
    gray_img = np.where(np.logical_and((gray_img < upper_thresh), (gray_img > lower_thresh)), 0, gray_img)
    gray_img[gray_img != 0] = 1
    return gray_img

def threshold_segment_color(img, lower_thresh, upper_thresh):
    """perform grayscale thresholding using a lower and upper threshold by
    blacking the background lying between the threholds and whitening the
    foreground
    Parameter
    ---------
    gray_img : ndarray
        grayscale image array
    lower_thresh : 3xndarray
        lowerbound to threshold (an intensity value between 0-255) (BGR)
    upper_thresh : 3xndarray
        upperbound to threshold (an intensity value between 0-255) (BGR)
    Returns
    -------
    ndarray
        thresholded version of gray_img
    """
    anded = np.logical_and((img < upper_thresh), (img > lower_thresh))
    alled = np.all(anded, axis=-1)
    return np.asarray(alled, dtype="uint8") * 255
    gray_img = img.mean(axis=-1)
    img = np.where(alled, 1, 0)
    img[img != 0] = 1
    return img

def do_kmeans(data, n_clusters):
    """Uses opencv to perform k-means clustering on the data given. Clusters it into
       n_clusters clusters.
       Args:
         data: ndarray of shape (n_datapoints, dim)
         n_clusters: int, number of clusters to divide into.
       Returns:
         clusters: integer array of length n_datapoints. clusters[i] is
         a number in range(n_clusters) specifying which cluster data[i]
         was assigned to. 
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, clusters, centers = kmeans = cv2.kmeans(data.astype(np.float32), n_clusters, bestLabels=None, criteria=criteria, attempts=1, flags=cv2.KMEANS_RANDOM_CENTERS)

    return clusters

def cluster_segment(img, n_clusters, random_state=0):
    """segment image using k_means clustering
    Parameter
    ---------
    img : ndarray
        rgb image array
    n_clusters : int
        the number of clusters to form as well as the number of centroids to generate
    random_state : int
        determines random number generation for centroid initialization
    Returns
    -------
    ndarray
        clusters of gray_img represented with similar pixel values
    """
    # First make image square by cropping the sides
    # height = img.shape[0]
    # width = img.shape[1]
    # diff = width - height
    # img = img[:, diff/2:-diff/2]
    # print("cropped image shape:", img.shape)

    # Downsample img by a factor of 2 first using the mean to speed up K-means
    img_d = cv2.resize(img, dsize=(img.shape[1]/2, img.shape[0]/2), interpolation=cv2.INTER_NEAREST)

    # TODO: Generate a clustered image using K-means

    # first convert our 3-dimensional img_d array to a 2-dimensional array
    # whose shape will be (height * width, number of channels) hint: use img_d.shape
    height = img_d.shape[0]
    width = img_d.shape[1]
    img_r = np.transpose(img_d.transpose(2,0,1).reshape(3,-1))
    
    # fit the k-means algorithm on this reshaped array img_r using the
    # the do_kmeans function defined above.
    clusters = do_kmeans(img_r, n_clusters)

    # reshape this clustered image to the original downsampled image (img_d) width and height 
    cluster_img = clusters.reshape(height, width)

    # Upsample the image back to the original image (img) using nearest interpolation
    img_u = cv2.resize(src=cluster_img, dsize=(img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    return img_u.astype(np.uint8)

def parse_args():
    """
    Parses arguments from the user. Read comments for more details.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-obj', type=str, default='cube', help=
        """Which Object you\'re trying to localize.  Options: cube, realsense, glass, plentea  
        Default: cube"""
    )
    # parser.add_argument('-n_vert', type=int, default=1000, help=
    #     'How many vertices you want to sample on the object surface.  Default: 1000'
    # )
    # parser.add_argument('-n_facets', type=int, default=32, help=
    #     """You will approximate the friction cone as a set of n_facets vectors along 
    #     the surface.  This way, to check if a vector is within the friction cone, all 
    #     you have to do is check if that vector can be represented by a POSITIVE 
    #     linear combination of the n_facets vectors.  Default: 32"""
    # )
    # parser.add_argument('-n_grasps', type=int, default=500, help=
    #     'How many grasps you want to sample.  Default: 500')
    # parser.add_argument('-n_execute', type=int, default=5, help=
    #     'How many grasps you want to execute.  Default: 5')
    # parser.add_argument('-metric', '-m', type=str, default='compute_force_closure', help=
    #     """Which grasp metric in grasp_metrics.py to use.  
    #     Options: compute_force_closure, compute_gravity_resistance, compute_robust_force_closure, compute_ferrari_canny"""
    # )
    # parser.add_argument('-arm', '-a', type=str, default='right', help=
    #     'Options: left, right.  Default: right'
    # )
    # parser.add_argument('-robot', type=str, default='baxter', help=
    #     """Which robot you're using.  Options: baxter, sawyer.  
    #     Default: baxter"""
    # )
    # parser.add_argument('--sim', action='store_true', help=
    #     """If you don\'t use this flag, you will only visualize the grasps.  This is 
    #     so you can run this outside of hte lab"""
    # )
    # parser.add_argument('--debug', action='store_true', help=
    #     'Whether or not to use a random seed'
    # )
    return parser.parse_args()

if __name__ == '__main__':

    rospy.init_node('dummy_tf_node')
    publisher = rospy.Publisher("visualization_marker_array", MarkerArray)
    args = parse_args()

    # for USB cam
    camera_topic = '/usb_cam/image_raw'
    camera_info = '/usb_cam/camera_info'
    camera_frame = '/usb_cam'

    # for RealSense cam
    realsense_pointcloud_topic = 'camera/depth/color/points'
    realsense_camera_frame = 'camera_depth_optical_frame'

    planner = PathPlanner('{}_arm'.format("right"))
    gripper = sawyer_gripper.Gripper("right")

    # Get initial mesh guess and generate particle filter
    #mesh = do_multiview(camera_topic, camera_info, camera_frame, planner, gripper)
    particle_filter = realsense_pointcloud(realsense_pointcloud_topic, realsense_camera_frame, obj_shape=args.obj)
    # Show mesh in vedo
    vedo.show(particle_filter.particles, new=True)
    markers = particle_filter.makeMarkers()
    print("MARKERS WROPJQTJKOWEJT==--=--=-=-=-=", markers)
    publisher.publish(markers)

    init_point = np.array([0.712, 0.168, -0.117+0.5])
    init_orientation = np.array([0, 1, 0, 0])

    # Show mesh in rviz
    while(True):
        sorted_particles = particle_filter.sorted_particles()
        best_particle = sorted_particles[0]
        #best_particle = particle_filter.center_particle()
        best_pose = best_particle.primitive.transform
        point = best_pose[:3, -1]
        #orientation = quaternion_from_matrix(best_pose[:3, :3])
        orientation = np.array([0,1,0,0])
        
        touch = moveit_plan(point, orientation, force_feedback=True, speed=0.2)
        curr_pose = lookup_transform("right_gripper_right_tip")
        particle_filter.reweight_particles(curr_pose, touch)
        particle_filter.renormalize_weights()
        particle_filter.sample_particles()


        x = raw_input("Start next loop? (Type something)")
        markers = particle_filter.makeMarkers()
        publisher.publish(markers)
        
        moveit_plan(init_point, init_orientation, force_feedback=False, speed=0.2)

    # args = parse_args()

    # if args.debug:
    #     np.random.seed(0)

    # if not args.sim:
    #     # Init rospy node (so we can use ROS commands)
    #     rospy.init_node('dummy_tf_node')

    # if args.obj != 'cube':
    #     # Mesh loading and pre-processing
    #     mesh = trimesh.load_mesh("objects/{}.obj".format(args.obj))
    #     # Transform object mesh to world frame
    #     T_world_obj = lookup_transform(args.obj, no_swap=True)
    #     print("DETECTED PAWN LOCATION", T_world_obj)
    #     mesh.apply_transform(T_world_obj)
    #     mesh.fix_normals()
    # else:
    #     camera_frame = ''
    #     if args.robot == 'baxter':
    #         camera_topic = '/cameras/left_hand_camera/camera_info'
    #         camera_info = '/cameras/left_hand_camera/camera_info'
    #         camera_frame = '/left_hand_camera'
    #     elif args.robot == 'sawyer':
    #         camera_topic = '/usb_cam/image_raw'
    #         camera_info = '/usb_cam/camera_info'
    #         camera_frame = '/usb_cam'
    #     else:
    #         print("Unknown robot type!")
    #         rospy.shutdown()
    #     mesh = locate_cube(camera_topic, camera_info, camera_frame)
        
    #     mesh.fix_normals()
    #     # pawn_mesh = trimesh.load_mesh("objects/{}.obj".format("pawn"))
    #     # # Transform object mesh to world frame
    #     # T_world_obj = lookup_transform("usb_cam")
    #     # T_world_obj[2, 3] + 0.2 
    #     # #print("DETECTED PAWN LOCATION", T_world_obj)
    #     # pawn_mesh.apply_transform(T_world_obj)
    #     # pawn_mesh.fix_normals()
    #     # vedo.show([mesh, pawn_mesh], new=True)

    # # This policy takes a mesh and returns the best actions to execute on the robot
    # grasping_policy = GraspingPolicy(
    #     args.n_vert, 
    #     args.n_grasps, 
    #     args.n_execute, 
    #     args.n_facets, 
    #     args.metric
    # )

    # # Each grasp is represented by T_grasp_world, a RigidTransform defining the 
    # # position of the end effector
    # grasp_vertices_total, grasp_poses = grasping_policy.top_n_actions(mesh, args.obj)

    # if not args.sim:
    #     # Execute each grasp on the baxter / sawyer
    #     if args.robot == "baxter":
    #         gripper = baxter_gripper.Gripper(args.arm)
    #         planner = PathPlanner('{}_arm'.format(args.arm))
    #     elif args.robot == "sawyer":
    #         gripper = sawyer_gripper.Gripper("right")
    #         planner = PathPlanner('{}_arm'.format("right"))
    #     else:
    #         print("Unknown robot type!")
    #         rospy.shutdown()

    # for grasp_vertices, grasp_pose in zip(grasp_vertices_total, grasp_poses):
    #     grasping_policy.visualize_grasp(mesh, grasp_vertices, grasp_pose)
    #     if not args.sim:
    #         repeat = True
    #         while repeat:
    #             execute_grasp(grasp_pose, planner, gripper)
    #             repeat = raw_input("repeat? [y|n] ") == 'y'
